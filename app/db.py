"""Postgres persistence (SQLAlchemy). Every run — single image, folder, or
test-folder eval — gets recorded here. The DB is queryable history on top of
the on-disk CSVs/images batch.py already writes; if Postgres is unreachable,
persistence is skipped with a warning rather than crashing the UI, since the
CSVs/annotated images on disk remain the source of truth for a given run.
"""

import contextlib
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://uav:uav@localhost:5433/uav")

_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mode: Mapped[str] = mapped_column(String(20))  # single | folder | test_folder
    source_path: Mapped[str | None] = mapped_column(String, nullable=True)
    output_path: Mapped[str | None] = mapped_column(String, nullable=True)
    pipeline_mode: Mapped[str] = mapped_column(String(64))
    confidence_cutoff: Mapped[float] = mapped_column(Float)
    iou_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    rfdetr_model: Mapped[str] = mapped_column(String(128))
    vlm_model: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    images: Mapped[list["Image"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    metrics: Mapped[list["RunMetric"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    undetected: Mapped[list["UndetectedClassSummary"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    filename: Mapped[str] = mapped_column(String)
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    modality: Mapped[str] = mapped_column(String(16))

    run: Mapped["Run"] = relationship(back_populates="images")
    ground_truths: Mapped[list["GroundTruth"]] = relationship(back_populates="image", cascade="all, delete-orphan")
    detections: Mapped[list["Detection"]] = relationship(back_populates="image", cascade="all, delete-orphan")


class GroundTruth(Base):
    __tablename__ = "ground_truths"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    class_name: Mapped[str] = mapped_column(String(32))
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)
    matched: Mapped[bool] = mapped_column(Boolean, default=False)

    image: Mapped["Image"] = relationship(back_populates="ground_truths")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    source: Mapped[str] = mapped_column(String(16))  # RFDETR | VLM
    class_name: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[int] = mapped_column(Integer)
    cascaded: Mapped[bool] = mapped_column(Boolean, default=False)
    matched_gt_id: Mapped[int | None] = mapped_column(ForeignKey("ground_truths.id"), nullable=True)
    iou: Mapped[float | None] = mapped_column(Float, nullable=True)
    match_result: Mapped[str | None] = mapped_column(String(8), nullable=True)  # TP | FP
    vlm_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    image: Mapped["Image"] = relationship(back_populates="detections")


class RunMetric(Base):
    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    scope: Mapped[str] = mapped_column(String(16))  # rfdetr | vlm | combined
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    accuracy: Mapped[float] = mapped_column(Float)
    mean_iou: Mapped[float] = mapped_column(Float)
    avg_time_per_image_ms: Mapped[float] = mapped_column(Float)
    images_per_sec: Mapped[float] = mapped_column(Float)

    run: Mapped["Run"] = relationship(back_populates="metrics")


class UndetectedClassSummary(Base):
    __tablename__ = "undetected_class_summary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    class_name: Mapped[str] = mapped_column(String(32))
    count: Mapped[int] = mapped_column(Integer)
    percentage: Mapped[float] = mapped_column(Float)

    run: Mapped["Run"] = relationship(back_populates="undetected")


def init_db() -> bool:
    """Create tables if they don't exist. Returns False (and doesn't raise) if Postgres is unreachable."""
    try:
        Base.metadata.create_all(_engine)
        return True
    except Exception:
        return False


@contextlib.contextmanager
def get_session():
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _add_detection(session, image: Image, det: dict, gt_row_by_id: dict | None = None) -> None:
    """matched_gt_id/iou/match_result come from the actual combined-scope IoU
    match (stashed as "_matched_gt_obj"/"_iou"/"_match_result" by save_eval_run's
    tp loop), not the mock's "_gt" simulation hint — those differ whenever the
    match was a wrong-class localization, and real (non-mock) detections never
    carry "_gt" at all."""
    matched_gt_id, iou_val, match_result = None, None, None
    if gt_row_by_id is not None:
        gt_obj = det.get("_matched_gt_obj")
        if gt_obj is not None and id(gt_obj) in gt_row_by_id:
            matched_gt_id = gt_row_by_id[id(gt_obj)].id
            iou_val = det.get("_iou")
            match_result = det.get("_match_result")

    session.add(Detection(
        image_id=image.id,
        source=det["source"],
        class_name=det["class_name"],
        confidence=det["confidence"],
        bbox_x=det["bbox"][0], bbox_y=det["bbox"][1], bbox_w=det["bbox"][2], bbox_h=det["bbox"][3],
        latency_ms=det.get("latency_ms", 0),
        cascaded=det.get("cascaded", False),
        matched_gt_id=matched_gt_id,
        iou=iou_val,
        match_result=match_result,
        vlm_payload={"reasoning": det["reasoning"]} if "reasoning" in det else None,
    ))


def save_single_image_run(filename: str, modality: str, rfdetr_model: str, vlm_model: str,
                           threshold: float, mode: str, cascade_result: dict,
                           output_path: str | None = None, annotated_path: str | None = None) -> None:
    try:
        with get_session() as session:
            run = Run(
                mode="single", source_path=filename, output_path=output_path,
                pipeline_mode=mode, confidence_cutoff=threshold,
                rfdetr_model=rfdetr_model, vlm_model=vlm_model,
            )
            session.add(run)
            session.flush()

            image = Image(run_id=run.id, filename=filename, path=annotated_path, modality=modality)
            session.add(image)
            session.flush()

            if cascade_result.get("rfdetr"):
                _add_detection(session, image, cascade_result["rfdetr"])
            if cascade_result.get("vlm"):
                v = dict(cascade_result["vlm"])
                v["cascaded"] = cascade_result.get("cascaded", False)
                _add_detection(session, image, v)
    except Exception:
        pass


def save_folder_run(folder_path: str, output_path: str, modality: str, rfdetr_model: str,
                     vlm_model: str, threshold: float, mode: str, per_image_results: list) -> None:
    """per_image_results: list of {"filename": str, "result": <run_cascade() dict>}"""
    try:
        with get_session() as session:
            run = Run(
                mode="folder", source_path=folder_path, output_path=output_path,
                pipeline_mode=mode, confidence_cutoff=threshold,
                rfdetr_model=rfdetr_model, vlm_model=vlm_model,
            )
            session.add(run)
            session.flush()

            for item in per_image_results:
                image = Image(run_id=run.id, filename=item["filename"],
                               path=item.get("annotated_path"), modality=modality)
                session.add(image)
                session.flush()

                result = item["result"]
                if result.get("rfdetr"):
                    _add_detection(session, image, result["rfdetr"])
                if result.get("vlm"):
                    v = dict(result["vlm"])
                    v["cascaded"] = result.get("cascaded", False)
                    _add_detection(session, image, v)
    except Exception:
        pass


def save_eval_run(test_folder_path: str, output_path: str, modality: str, rfdetr_model: str,
                   vlm_model: str, threshold: float, mode: str, iou_threshold: float,
                   per_image_results: list, scope_metrics: dict, undetected_breakdown: dict,
                   undetected_counts: dict) -> None:
    """
    per_image_results: list of {"filename": str, "gt_boxes": [GTBox], "combined": [detection dict],
                                 "combined_match": metrics.MatchResult, "time_ms": float}
    scope_metrics: {"rfdetr": {...}, "vlm": {...}, "combined": {...}} from metrics.compute_scope_metrics
    """
    try:
        with get_session() as session:
            run = Run(
                mode="test_folder", source_path=test_folder_path, output_path=output_path,
                pipeline_mode=mode, confidence_cutoff=threshold, iou_threshold=iou_threshold,
                rfdetr_model=rfdetr_model, vlm_model=vlm_model,
            )
            session.add(run)
            session.flush()

            for item in per_image_results:
                image = Image(run_id=run.id, filename=item["filename"],
                               path=item.get("annotated_path"), modality=modality)
                session.add(image)
                session.flush()

                gt_row_by_id = {}
                for gt in item["gt_boxes"]:
                    gt_row = GroundTruth(
                        image_id=image.id, class_name=gt.class_name,
                        bbox_x=gt.bbox[0], bbox_y=gt.bbox[1], bbox_w=gt.bbox[2], bbox_h=gt.bbox[3],
                    )
                    session.add(gt_row)
                    gt_row_by_id[id(gt)] = gt_row
                session.flush()

                match = item.get("combined_match")
                if match is not None:
                    for gt, _pred in [(g, None) for g in match.fn]:
                        row = gt_row_by_id.get(id(gt))
                        if row is not None:
                            row.matched = False
                    for pred, gt, iou_val in match.tp:
                        pred["_iou"] = iou_val
                        pred["_match_result"] = "TP"
                        pred["_matched_gt_obj"] = gt
                        row = gt_row_by_id.get(id(gt))
                        if row is not None:
                            row.matched = True

                for det in item["combined"]:
                    _add_detection(session, image, det, gt_row_by_id)

            for scope, m in scope_metrics.items():
                session.add(RunMetric(
                    run_id=run.id, scope=scope,
                    precision=m["precision"], recall=m["recall"], f1=m["f1"],
                    accuracy=m["accuracy"], mean_iou=m["mean_iou"],
                    avg_time_per_image_ms=m.get("avg_time_per_image_ms", 0.0),
                    images_per_sec=m.get("images_per_sec", 0.0),
                ))

            for class_name, pct in undetected_breakdown.items():
                session.add(UndetectedClassSummary(
                    run_id=run.id, class_name=class_name,
                    count=undetected_counts.get(class_name, 0), percentage=pct,
                ))
    except Exception:
        pass


def list_runs(limit: int = 50) -> list[dict]:
    """Most-recent-first summary of past runs, for the History tab's run picker."""
    try:
        with get_session() as session:
            runs = session.query(Run).order_by(Run.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": r.id, "mode": r.mode, "created_at": r.created_at,
                    "pipeline_mode": r.pipeline_mode, "rfdetr_model": r.rfdetr_model,
                    "vlm_model": r.vlm_model, "confidence_cutoff": r.confidence_cutoff,
                    "iou_threshold": r.iou_threshold, "source_path": r.source_path,
                    "output_path": r.output_path, "num_images": len(r.images),
                }
                for r in runs
            ]
    except Exception:
        return []


def get_run_detail(run_id: int) -> dict | None:
    """Full detail for one run — images/detections/ground-truths/metrics/undetected
    breakdown — for the History tab's drill-down view."""
    try:
        with get_session() as session:
            run = session.get(Run, run_id)
            if run is None:
                return None

            images = []
            for img in run.images:
                detections = [
                    {
                        "source": d.source, "class_name": d.class_name, "confidence": d.confidence,
                        "bbox": (d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h), "latency_ms": d.latency_ms,
                        "cascaded": d.cascaded, "iou": d.iou, "match_result": d.match_result,
                        "reasoning": (d.vlm_payload or {}).get("reasoning"),
                    }
                    for d in img.detections
                ]
                ground_truths = [
                    {"class_name": g.class_name, "bbox": (g.bbox_x, g.bbox_y, g.bbox_w, g.bbox_h),
                     "matched": g.matched}
                    for g in img.ground_truths
                ]
                images.append({
                    "filename": img.filename, "path": img.path, "modality": img.modality,
                    "detections": detections, "ground_truths": ground_truths,
                })

            metrics = [
                {"scope": m.scope, "precision": m.precision, "recall": m.recall, "f1": m.f1,
                 "accuracy": m.accuracy, "mean_iou": m.mean_iou,
                 "avg_time_per_image_ms": m.avg_time_per_image_ms, "images_per_sec": m.images_per_sec}
                for m in run.metrics
            ]
            undetected = [
                {"class_name": u.class_name, "count": u.count, "percentage": u.percentage}
                for u in run.undetected
            ]

            return {
                "id": run.id, "mode": run.mode, "created_at": run.created_at,
                "pipeline_mode": run.pipeline_mode, "rfdetr_model": run.rfdetr_model,
                "vlm_model": run.vlm_model, "confidence_cutoff": run.confidence_cutoff,
                "iou_threshold": run.iou_threshold, "source_path": run.source_path,
                "output_path": run.output_path, "images": images, "metrics": metrics,
                "undetected": undetected,
            }
    except Exception:
        return None

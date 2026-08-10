"""Postgres persistence (SQLAlchemy). Every run — single image or folder —
gets recorded here. The DB is queryable history on top of the on-disk
CSVs/images batch.py already writes; if Postgres is unreachable,
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
    mode: Mapped[str] = mapped_column(String(20))  # single | folder
    source_path: Mapped[str | None] = mapped_column(String, nullable=True)
    output_path: Mapped[str | None] = mapped_column(String, nullable=True)
    pipeline_mode: Mapped[str] = mapped_column(String(64))
    confidence_cutoff: Mapped[float] = mapped_column(Float)
    rfdetr_model: Mapped[str] = mapped_column(String(128))
    vlm_model: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    images: Mapped[list["Image"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    filename: Mapped[str] = mapped_column(String)
    path: Mapped[str | None] = mapped_column(String, nullable=True)
    modality: Mapped[str] = mapped_column(String(16))

    run: Mapped["Run"] = relationship(back_populates="images")
    detections: Mapped[list["Detection"]] = relationship(back_populates="image", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    source: Mapped[str] = mapped_column(String(16))  # YOLO | VLM
    class_name: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    bbox_x: Mapped[float] = mapped_column(Float)
    bbox_y: Mapped[float] = mapped_column(Float)
    bbox_w: Mapped[float] = mapped_column(Float)
    bbox_h: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[int] = mapped_column(Integer)
    cascaded: Mapped[bool] = mapped_column(Boolean, default=False)
    vlm_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    image: Mapped["Image"] = relationship(back_populates="detections")


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


def _add_detection(session, image: Image, det: dict) -> None:
    session.add(Detection(
        image_id=image.id,
        source=det["source"],
        class_name=det["class_name"],
        confidence=det["confidence"],
        bbox_x=det["bbox"][0], bbox_y=det["bbox"][1], bbox_w=det["bbox"][2], bbox_h=det["bbox"][3],
        latency_ms=det.get("latency_ms", 0),
        cascaded=det.get("cascaded", False),
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

            for det in cascade_result.get("rfdetr_all") or ([cascade_result["rfdetr"]] if cascade_result.get("rfdetr") else []):
                _add_detection(session, image, det)
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
                for det in result.get("rfdetr_all") or ([result["rfdetr"]] if result.get("rfdetr") else []):
                    _add_detection(session, image, det)
                if result.get("vlm"):
                    v = dict(result["vlm"])
                    v["cascaded"] = result.get("cascaded", False)
                    _add_detection(session, image, v)
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
                    "source_path": r.source_path,
                    "output_path": r.output_path, "num_images": len(r.images),
                }
                for r in runs
            ]
    except Exception:
        return []


def get_run_detail(run_id: int) -> dict | None:
    """Full detail for one run — images/detections — for the History tab's
    drill-down view."""
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
                        "cascaded": d.cascaded,
                        "reasoning": (d.vlm_payload or {}).get("reasoning"),
                    }
                    for d in img.detections
                ]
                images.append({
                    "filename": img.filename, "path": img.path, "modality": img.modality,
                    "detections": detections,
                })

            return {
                "id": run.id, "mode": run.mode, "created_at": run.created_at,
                "pipeline_mode": run.pipeline_mode, "rfdetr_model": run.rfdetr_model,
                "vlm_model": run.vlm_model, "confidence_cutoff": run.confidence_cutoff,
                "source_path": run.source_path, "output_path": run.output_path, "images": images,
            }
    except Exception:
        return None

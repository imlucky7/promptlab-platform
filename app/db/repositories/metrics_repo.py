"""Repository for the ``metrics_logs`` collection."""

from __future__ import annotations

from app.db.repositories.base_repo import BaseRepository


class MetricsRepository(BaseRepository):
    """Data access for derived metrics logs."""

    collection_name = "metrics_logs"

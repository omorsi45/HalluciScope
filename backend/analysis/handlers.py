"""The run_analysis job and its failure hook.

The job payload is the analysis id. The question and answer live only in the RLS table
`analyses`, and nothing in a log line or job row carries them.
"""

import logging
from uuid import UUID

from backend.analysis.service import run_analysis
from backend.jobs.worker import JobContext

logger = logging.getLogger(__name__)


async def handle_run_analysis(ctx: JobContext) -> None:
    await run_analysis(ctx.services, ctx.job.tenant_id, UUID(ctx.job.payload["analysis_id"]))


async def fail_run_analysis(ctx: JobContext, error_code: str) -> None:
    """Failure hook: a job that died (retries used up, lease expired) fails its analysis.

    An analysis that already succeeded or failed is left as it is, so a second call changes
    nothing. Never raises: the worker must be able to settle other jobs after it.
    """
    try:
        analysis_id = UUID(ctx.job.payload["analysis_id"])
        async with ctx.db.tenant_tx(ctx.job.tenant_id) as conn:
            await conn.execute(
                "update analyses set status = 'failed', error_code = %s, completed_at = now() "
                "where id = %s and status in ('queued', 'running')",
                (error_code, analysis_id),
            )
    except Exception:
        logger.exception("analysis failure hook error")

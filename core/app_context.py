import logging

from agents.resume_agent import ResumeAgent
from database import redis_client
from agents.interview_agent import InterviewAgent


agent = ResumeAgent(redis_client)
interview_agent = InterviewAgent(agent=agent, redis_client=redis_client)
logger = logging.getLogger(__name__)

PLAYGROUND_ENABLED = True
OBSERVABILITY_ENABLED = True



from learning.ppo.ppo_agent import construct_agent as ppo_construct
from learning.sac.sac_agent import construct_agent as sac_construct


AGENT_CONSTRUCT_FUNC = {
    "ppo": ppo_construct,
    "sac": sac_construct
}
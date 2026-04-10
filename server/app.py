from openenv.core import create_app
from sqloptimenv.environment import SQLOptimEnv, SQLAction, SQLObservation

app = create_app(
    env=SQLOptimEnv,
    action_cls=SQLAction,
    observation_cls=SQLObservation,
    env_name="sqloptimenv",
)
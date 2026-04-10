from openenv.core import create_app
from sqloptimenv.environment import SQLOptimEnv, SQLAction, SQLObservation
import uvicorn

app = create_app(
    env=SQLOptimEnv,
    action_cls=SQLAction,
    observation_cls=SQLObservation,
    env_name="sqloptimenv",
)

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
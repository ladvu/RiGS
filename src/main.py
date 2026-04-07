from config import Config
import tyro
from runner import *
from utils.misc_utils import seed_everything

# TODO: integrate viewer

def main(cfg: Config):
    runner = {
        "fused": FusedRunner,
        "dynamic": DynamicRunner,
        "static": StaticRunner,
    }[cfg.mode](cfg)

    if cfg.ckpt is not None:
        # run eval only
        res = runner.load_checkpoint(cfg.ckpt)
        print(res)
        if cfg.test_time_pose_opt:
            runner.pose_align(cfg.eval_step)
        if cfg.render_video:
            runner.render_video()
        elif cfg.render_point:
            runner.render_point()
        elif cfg.eval_step is not None:
            runner.eval(cfg.eval_step)
    else:
        seed_everything(cfg.seed)
        runner.train()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
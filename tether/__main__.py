"""Command line entry point. Run `python -m tether --help`."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tether", description="Supermix Tether CPU research laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    study = sub.add_parser("study", help="train a fresh reproducible study; refuses overwrite")
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--seeds", default="11,12,13")
    study.add_argument("--arms", default="passive,dropout,iit,effect_only,invariance_only,unpaired,tether")
    study.add_argument("--steps", type=int, default=400)
    study.add_argument("--donor-steps", type=int, default=180)
    study.add_argument("--batch-size", type=int, default=128)
    study.add_argument("--eval-size", type=int, default=512)
    study.add_argument("--validation-every", type=int, default=50)
    study.add_argument("--learning-rate", type=float, default=.02)
    study.add_argument("--effect-weight", type=float, default=1.)
    study.add_argument("--invariance-weight", type=float, default=1.)
    study.add_argument("--threads", type=int, default=2)
    study.add_argument("--phase", choices=("development", "confirmation"), default="development")
    replay = sub.add_parser("replay", help="recompute saved metrics and optionally checkpoint inference")
    replay.add_argument("study", type=Path)
    replay.add_argument("--strict", action="store_true")
    serve = sub.add_parser("serve", help="open a read-only local results service")
    serve.add_argument("study", type=Path)
    serve.add_argument("--port", type=int, default=8797)
    args = parser.parse_args(argv)
    if args.command == "study":
        from .experiment import StudyConfig, run_study
        config = StudyConfig(seeds=tuple(int(s) for s in args.seeds.split(",")), arms=tuple(args.arms.split(",")),
                             steps=args.steps, donor_steps=args.donor_steps, batch_size=args.batch_size,
                             eval_size=args.eval_size, validation_every=args.validation_every, learning_rate=args.learning_rate,
                             effect_weight=args.effect_weight, invariance_weight=args.invariance_weight, threads=args.threads, phase=args.phase)
        result = run_study(config, args.output, progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps({"output": str(args.output.resolve()), "decision": result["decision"]}, indent=2))
    elif args.command == "replay":
        from .experiment import replay_study
        print(json.dumps(replay_study(args.study, strict=args.strict), indent=2))
    elif args.command == "serve":
        from .server import serve
        serve(args.study, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

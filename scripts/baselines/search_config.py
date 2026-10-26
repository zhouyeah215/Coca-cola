"""Resolve search credentials before loading any model."""
import os


def resolve_search_args(parser, args):
    provider_env = {"exa": "EXA_API_KEY", "tavily": "TAVILY_API_KEY"}[args.search_provider]
    if args.search_api_key is None:
        args.search_api_key = os.environ.get("SEARCH_API_KEY") or os.environ.get(provider_env)
    if not args.search_api_key or not args.search_api_key.strip():
        parser.error(f"{args.search_provider} requires --search_api_key, SEARCH_API_KEY or {provider_env}")
    return args

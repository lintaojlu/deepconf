"""Dynasor OpenAI client API example"""

import argparse
from openai import OpenAI

import logging

def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger

logger = init_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI Chat Client")
    parser.add_argument("--api-key", default="EMPTY", help="OpenAI API key")
    parser.add_argument(
        "--base-url", default="http://localhost:8001/v1", help="OpenAI API base URL"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens for completion"
    )
    parser.add_argument(
        "--saving-effort",
        default="mid",
        choices=["none", "mild", "low", "mid", "high", "crazy"],
        help="Dynasor saving effort level",
    )
    parser.add_argument("--prompt", default="2+2=", help="User prompt")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.debug("Args: %s", args)

    stream = args.stream

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=1,
    )

    logger.debug("Grab models")
    models = client.models.list()
    model = models.data[0].id
    logger.debug("Model: %s", model)

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
        model=model,
        max_tokens=args.max_tokens,
        extra_body={"dynasor": {"saving_effort": args.saving_effort}},
        stream=stream,
    )

    
    print("Prompt: ", args.prompt)
    print("-" * 10)
    print("Response: \n")
    if not stream:
        print(response.choices[0].message.content)
        return

    else:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)

    # for chunk in response:
    #     if chunk.choices[0].delta.content:  # Only print non-empty content chunks
    #         print(chunk.choices[0].delta.content, end="", flush=True)

    # for chunk in response:
    #     print(chunk.choices[0].delta.content, end="", flush=True)

    # print("Chat completion results:")
    # print(chat_completion)


if __name__ == "__main__":
    main()

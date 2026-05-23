#!/usr/bin/env python
"""
Download from W&B and apply basic data cleaning and then export the result to new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()


    local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)
    logger.info(f"data version: {run.use_artifact(args.input_artifact).version}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    artifact.wait()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Path to the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Path to the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="cleaned data artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price for filtering the data",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price for filtering the data",
        required=True
    )


    args = parser.parse_args()

    go(args)

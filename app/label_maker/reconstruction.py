#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import misc3d as m3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to pipeline config file')
    args = parser.parse_args()

    pipeline = m3d.reconstruction.ReconstructionPipeline(args.config)
    pipeline.run_system()

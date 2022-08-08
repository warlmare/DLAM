#!/bin/bash

find dataset/anomalies/ -name '*' | xargs rm
rm -rf dataset/anomalies/*
rm -rf dataset/anomaly

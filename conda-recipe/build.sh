#!/bin/bash
$PYTHON setup.py install --single-version-externally-managed --record=record.txt

mkdir $PREFIX/streamvis-apps
cp -r apps/* $PREFIX/streamvis-apps
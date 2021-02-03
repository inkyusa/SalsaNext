#!/bin/sh

#======================================================
#  All weights will be stored under this folder
#======================================================
export ModelPath=models

#======================================================
#  Variables definitions
#======================================================
RELEASE_VERSION=v0.1
GIT_BASE_URL=https://github.com/inkyusa/SalsaNext/releases/download/${RELEASE_VERSION}
PRETRAINED=pretrained
FIRST_TRAINED=first_trained
PRETRAINED_URL=${GIT_BASE_URL}/${PRETRAINED}.zip
FIRST_TRAINED_URL=${GIT_BASE_URL}/${FIRST_TRAINED}.zip


#======================================================
#  Creating a folder to save downloaded models
#======================================================
if [ ! -d "${ModelPath}" ]; then
  echo "=================================="
  echo "   Creating ${ModelPath} folder  "
  echo "=================================="
  mkdir ${ModelPath}
fi

#======================================================
#  Downloading pretrained (author provided) weights
#======================================================

if [ ! -d "${ModelPath}/${PRETRAINED}" ]; then
  if [ ! -f "${ModelPath}/${PRETRAINED}" ]; then
    echo "=================================="
    echo "  Downloading ${PRETRAINED}.zip to ${ModelPath}/${PRETRAINED}"
    echo "=================================="
    wget ${PRETRAINED_URL} -P ${ModelPath}
  fi
  echo "=================================="
  echo "  Uncompressing ${PRETRAINED}.zip"
  echo "=================================="
  unzip -d ${ModelPath} "${ModelPath}/${PRETRAINED}"
  echo "=================================="
  echo "  Delete ${PRETRAINED}.zip"
  echo "=================================="
  rm "${ModelPath}/${PRETRAINED}.zip"
fi

#======================================================
#  Downloading first trained weights (150 epoches)
#======================================================
if [ ! -d "${ModelPath}/${FIRST_TRAINED}" ]; then
  if [ ! -f "${ModelPath}/${FIRST_TRAINED}" ]; then
    echo "=================================="
    echo "  Downloading ${FIRST_TRAINED}.zip to ${ModelPath}/${FIRST_TRAINED}"
    echo "=================================="
    wget ${FIRST_TRAINED_URL} -P ${ModelPath}
  fi
  echo "=================================="
  echo "  Uncompressing ${FIRST_TRAINED}.zip"
  echo "=================================="
  unzip -d ${ModelPath} "${ModelPath}/${FIRST_TRAINED}.zip"
  echo "=================================="
  echo "  Delete ${FIRST_TRAINED}.zip"
  echo "=================================="
  rm "${ModelPath}/${FIRST_TRAINED}.zip"
fi
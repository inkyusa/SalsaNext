#!/bin/sh

export ModelPath=models

PRETRAINED=https://github.com/inkyusa/SalsaNext/releases/download/v0.1/pretrained.zip
FIRST_TRAINED=https://github.com/inkyusa/SalsaNext/releases/download/v0.1/first_trained_model.zip



if [ ! -d "${ModelPath}" ]; then
  echo "=================================="
  echo "   Creating ${ModelPath} folder  "
  echo "=================================="
  mkdir ${ModelPath}
fi


if [ ! -d "${ModelPath}/pretrained" ]; then
  if [ ! -f "${PRETRAINED" ]; then
    echo "=================================="
    echo "  Downloading ${ROBERTA_BASE} to ${DownloadPath}"
    echo "=================================="
    wget ${ROBERTA_BASE} -P ${DownloadPath}
  fi
  echo "=================================="
  echo "  Uncompressing ${ROBERTA_BASE}"
  echo "=================================="
  unzip -d ${DownloadPath} "${DownloadPath}/${RobertaBase}.zip"
  echo "=================================="
  echo "  Delete ${RobertaBase}.zip"
  echo "=================================="
  rm "${DownloadPath}/${RobertaBase}.zip"
fi

if [ ! -d "input/roberta-base" ]; then
  if [ ! -f "${DownloadPath}/${RobertaBase}.zip" ]; then
    echo "=================================="
    echo "  Downloading ${ROBERTA_BASE} to ${DownloadPath}"
    echo "=================================="
    wget ${ROBERTA_BASE} -P ${DownloadPath}
  fi
  echo "=================================="
  echo "  Uncompressing ${ROBERTA_BASE}"
  echo "=================================="
  unzip -d ${DownloadPath} "${DownloadPath}/${RobertaBase}.zip"
  echo "=================================="
  echo "  Delete ${RobertaBase}.zip"
  echo "=================================="
  rm "${DownloadPath}/${RobertaBase}.zip"
fi

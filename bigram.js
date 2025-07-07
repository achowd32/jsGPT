#!/usr/bin/env node

const fs = require('fs');

// hyperparameters
BLOCK_SIZE = 8;

// read in data file
const dataStr = fs.readFileSync('data.txt').toString();
const splitInd = Math.round(0.9 * dataStr.length);
const trainStr = dataStr.slice(0, splitInd);
const valStr = dataStr.slice(splitInd);

// set up token encoder and decoder
const charList = Array.from(new Set(dataStr)).sort();
const stoi = new Map(charList.map((val, ind) => [val, ind]));
const itos = new Map(charList.map((val, ind) => [ind, val]));
function encode(str){
  return Array.from(str).map(c => stoi.get(c));
}
function decode(toks){
  return toks.map(i => itos.get(i)).join('');
}



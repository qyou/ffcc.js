/**
 * ffcc.js
 * 
 * 
 * Revised from https://github.com/yuanxy92/AutoWhiteBalance
 * model.json is extracted from model.mat.
 * 
 * See the following paper in detail:
 *    Barron, Jonathan T., and Yun-Ta Tsai. "Fast Fourier Color Constancy." arXiv preprint arXiv:1611.07596 (2016).
 */

const cv = require('opencv4nodejs')
const {
  f,
  b
} = require('./model.json')


const uv0 = -1.421875
const binSize = 1 / 64
const binNum = 256
const eps = 1e-7

function calcLogHist(image) {
  const {
    rows,
    cols
  } = image
  const imageLog = image.convertTo(cv.CV_32FC3).log()
  const [bMat, gMat, rMat] = imageLog.splitChannels()
  const uMat = gMat.sub(rMat)
  const vMat = gMat.sub(bMat)
  const hist = new cv.Mat(binNum, binNum, cv.CV_32FC1, 0)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; ++j) {
      const u = uMat.atRaw(i, j)
      const v = vMat.atRaw(i, j)
      if (Number.isFinite(u) && Number.isFinite(v) && !Number.isNaN(u) && !Number.isNaN(v)) {
        const uVal = Math.max(Math.min(Math.round((u - uv0) / binSize), 256), 1)
        const vVal = Math.max(Math.min(Math.round((v - uv0) / binSize), 256), 1)
        hist.set(uVal - 1, vVal - 1, hist.atRaw(uVal - 1, vVal - 1) + 1)
      }
    }
  }
  const sum = hist.sum()
  return hist.div(Math.max(eps, sum))
}

function loadModel() {
  const fMat = new cv.Mat(f, cv.CV_32FC1)
  const bMat = new cv.Mat(b, cv.CV_32FC1)
  return {
    fMat,
    bMat
  }
}

// use dft and inverse-dft
function applyModel(hist, {
  fMat,
  bMat
}) {
  const histDFTMat = hist.dft(cv.DFT_COMPLEX_OUTPUT)
  const weightDFTMat = fMat.dft(cv.DFT_COMPLEX_OUTPUT)
  const biasDFTMat = bMat
    .dft(cv.DFT_COMPLEX_OUTPUT)
    .div(2)

  const outDFTMat = histDFTMat.mulSpectrums(weightDFTMat).add(biasDFTMat)
  // reverse DFT and get real out
  const outMat = outDFTMat.dft(cv.DFT_INVERSE + cv.DFT_REAL_OUTPUT)
  const outRealMat = outMat.splitChannels()[0]
  // get the max element location
  const {
    maxLoc
  } = outRealMat.minMaxLoc()
  const u = (maxLoc.y + 1) * binSize + uv0
  const v = (maxLoc.x + 1) * binSize + uv0
  return {
    responseMat: outRealMat,
    u,
    v
  }
}

function applyWhiteBalance(image, u, v) {
  const expNegU = Math.exp(-u)
  const expNegV = Math.exp(-v)
  const z = Math.sqrt(expNegU * expNegU + expNegV * expNegV + 1)
  const r = expNegU / z
  const g = 1 / z
  const b = expNegV / z
  const imageF = image.convertTo(cv.CV_32FC3)
  let [bMat,
    gMat,
    rMat
  ] = imageF.splitChannels()
  bMat = bMat.div(b)
  gMat = gMat.div(g)
  rMat = rMat.div(r)
  const mergeMat = new cv.Mat([bMat, gMat, rMat])
  const normalizedMat = mergeMat.normalize(1, 0, cv.NORM_MINMAX)
  const resultMat = normalizedMat.convertTo(cv.CV_8U, 255, 0)
  return resultMat
}

// enhance the input mat by using ffcc (color consistency)
function enhance(image) {
  const hist = calcLogHist(image)
  const {
    u,
    v
  } = applyModel(hist, loadModel())
  return applyWhiteBalance(image, u, v)
}


module.exports = {
  loadModel,
  calcLogHist,
  applyModel,
  applyWhiteBalance,
  enhance
}
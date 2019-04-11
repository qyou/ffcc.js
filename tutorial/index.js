const { enhance } = require('../src/ffcc')
const cv = require('opencv4nodejs')

function main() {
  const imagePath = '000002.png'
  const image = cv.imread(imagePath)
  cv.imshow('original', image)
  const enhancedImage = enhance(image)
  cv.imshowWait('enhanced', enhancedImage)
}

main()
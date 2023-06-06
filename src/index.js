import '../node_modules/materialize-css/dist/js/materialize.min.js';
import './styles.css';

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

// globals
let model;
let normalisedFeature;
let normalisedLabel;
let trainingFeatures;
let testingFeatures;
let trainingLabels;
let testingLabels;
const localStorageID = 'house-price-regression';
const points = [];

// utils
const plot = (points, featureName, predictedPoints = null) => {
  const values = [points];
  const series = ['original'];

  if (Array.isArray(predictedPoints)) {
    values.push(predictedPoints);
    series.push('prediceted');
  }

  tfvis.render.scatterplot(
      {name: `${featureName} vs House Price`}, {values, series}, {
        xLabel: featureName,
        yLabel: 'Price',
      });
};
const plotPrediction = async () => {
  const [xs, ys] = tf.tidy(() => {
    const normalisedXs =
        tf.linspace(0, 1, 100); // range of 100 points within 0 and 1
    const normalisedYs = model.predict(
        normalisedXs.reshape([100, 1])); // predict expects 2d tensor

    const xs =
        denormalise(normalisedXs, normalisedFeature.min, normalisedFeature.max);
    const ys =
        denormalise(normalisedYs, normalisedLabel.min, normalisedLabel.max);

    return [xs.dataSync(), ys.dataSync()];
  });

  console.log('XS', xs);
  console.log('YS', ys);

  const predictedPoints = Array.from(xs).map((x, i) => ({x, y: ys[i]}));

  await plot(points, 'Square feet', predictedPoints);
};
const normalise = (tensor, previousMin = null, previousMax = null) => {
  const max = previousMax ?? tensor.max();
  const min = previousMin ?? tensor.min();
  const normalisedTensor = tensor.sub(min).div(max.sub(min));

  return {tensor: normalisedTensor, min, max};
};
const denormalise = (tensor, min, max) => {
  return tensor.mul(max.sub(min)).add(min);
};
const trainModel =
    async (model, trainingFeatureTensor, trainingLabelTensor) => {
      const {onEpochEnd} = tfvis.show.fitCallbacks(
          {
            name: 'Training Performance',
          },
          ['loss']);

      return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 20,
        callbacks: {
          onEpochBegin: async () => {
            await plotPrediction();
            // update layer summary showing current values of the weights
            const layer = model.getLayer(undefined, 0); // 1st and only layer
            tfvis.show.layer({name: 'Layer'}, layer);
          },
          onEpochEnd,
        },
      });
    };

const createModel = () => {
  model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1, // single node
    useBias: true,
    activation: 'linear', // no threshold
    inputDim: 1,
  }));

  return model;
};

const run = async () => {
  await tf.ready();
  const houseSalesDataset =
      tf.data.csv('http://127.0.0.1:3000/kc_house_data.csv');
  const pointsDataset = houseSalesDataset.map((record) => ({
    x: record.sqft_living,
    y: record.price,
  }));

  await pointsDataset.forEachAsync((point) => {
    points.push(point);
  });

  if (points.length % 2 !== 0) {
    points.pop();
  }

  const featureValues = points.map((p) => p.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);
  const labelValues = points.map((p) => p.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  normalisedFeature = normalise(featureTensor);
  normalisedLabel = normalise(labelTensor);

  const trainingSize = Math.round(normalisedFeature.tensor.shape[0] * 0.8);
  const testingSize = Math.round(normalisedFeature.tensor.shape[0] * 0.2);

  [trainingFeatures, testingFeatures] = tf.split(normalisedFeature.tensor, [
    trainingSize,
    testingSize,
  ]);
  [trainingLabels, testingLabels] = tf.split(normalisedLabel.tensor, [
    trainingSize,
    testingSize,
  ]);

  // update state and enable train button
  document.querySelector('#model-status').innerHTML = 'No model trained';
  document.querySelector('#train-button').removeAttribute('disabled');
  document.querySelector('#load-button').removeAttribute('disabled');
};

export const predict = async () => {
  const predictionInput =
      parseInt(document.getElementById('prediction-input').value);
  if (isNaN(predictionInput)) {
    alert('Please enter a valid number');
  } else {
    tf.tidy(() => {
      const inputTensor = tf.tensor1d([predictionInput]);
      const normalisedInput =
          normalise(inputTensor, normalisedFeature.min, normalisedFeature.max);
      const normalisedOutputTensor = model.predict(normalisedInput.tensor);
      const outputTensor = denormalise(
          normalisedOutputTensor, normalisedLabel.min, normalisedLabel.max);
      const outputValue = outputTensor.dataSync()[0];
      const outputValueRounded = Math.round(outputValue / 1000) * 1000;
      document.getElementById('prediction-output').innerHTML =
          `$${outputValueRounded}`;
    });
  }
};

export const load = async () => {
  const storageKey = `localstorage://${localStorageID}`;
  const models = await tf.io.listModels();
  const modelInfo = models[storageKey];
  if (modelInfo) {
    model = await tf.loadLayersModel(storageKey);

    // show model on visor
    const layer = model.getLayer(undefined, 0); // 1st and only layer
    tfvis.show.modelSummary({name: `Model Summary`, tab: `Model`}, model);
    tfvis.show.layer({name: 'Layer'}, layer);

    await plotPrediction();

    // update statuses
    document.querySelector('#model-status').innerHTML =
        `Trained (saved ${modelInfo.dateSaved})`;
    document.querySelector('#test-button').removeAttribute('disabled');
    document.querySelector('#predict-button').removeAttribute('disabled');
  } else {
    alert('Could not load: no model found');
  }
};

export const save = async () => {
  const saveResults = await model.save(`localstorage://${localStorageID}`);
  document.querySelector('#model-status').innerHTML =
      `Trained (saved ${saveResults.modelArtifactsInfo.dateSaved})`;
};

export const test = async () => {
  const lossTensor = model.evaluate(testingFeatures, testingLabels);
  const loss = await lossTensor.dataSync(); // data sync returns numeric value

  document.querySelector('#testing-status').innerHTML =
      `Testing set loss: ${Number(loss).toPrecision(5)}`;
};

export const train = async () => {
  // disable all buttons
  ['train', 'test', 'load', 'predict', 'save'].forEach((id) => {
    document.querySelector(`#${id}-button`)
        ?.setAttribute('disabled', 'disabled');
  });
  document.querySelector('#model-status').innerHTML = 'Training...';

  model = createModel();

  model.summary();
  const layer = model.getLayer(undefined, 0); // 1st and only layer
  tfvis.show.modelSummary({name: `Model Summary`, tab: `Model`}, model);
  tfvis.show.layer({name: 'Layer'}, layer);

  const optimizer = tf.train.sgd(0.1);

  model.compile({
    loss: 'meanSquaredError',
    optimizer,
  });

  await plotPrediction();

  const result = await trainModel(model, trainingFeatures, trainingLabels);
  const trainingLoss = result.history.loss.pop();

  document.querySelector('#model-status').innerHTML = 'Trained (unsaved) \n' +
      `Loss ${trainingLoss.toPrecision(5)}\n`;

  document.querySelector('#test-button')?.removeAttribute('disabled');
  document.querySelector('#save-button')?.removeAttribute('disabled');
};

export const toggleVisor = async () => {
  tfvis.visor().toggle();
};

tf.tidy(() => {
  run();
});

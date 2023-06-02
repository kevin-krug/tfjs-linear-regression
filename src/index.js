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

// utils
const plot = (points, featureName) => {
  tfvis.render.scatterplot(
      {name: `${featureName} vs House Price`},
      {values: [points], series: ['original']},
      {
        xLabel: featureName,
        yLabel: 'Price',
      },
  );
};
const normalise = (tensor) => {
  const max = tensor.max();
  const min = tensor.min();
  const normalisedTensor = tensor.sub(min).div(max.sub(min));

  return {tensor: normalisedTensor, min, max};
};
const denormalise = (tensor, min, max) => {
  return tensor.mul(max.sub(min)).add(min);
};
const trainModel =
    async (model, trainingFeatureTensor, trainingLabelTensor) => {
      const {onBatchEnd, onEpochEnd} = tfvis.show.fitCallbacks(
          {
            name: 'Training Performance',
          },
          ['loss']);

      return model.fit(trainingFeatureTensor, trainingLabelTensor, {
        batchSize: 32,
        epochs: 20,
        callbacks: {
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
  const points = [];

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
  plot(points, 'Square Feet');

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
  document.querySelector('#train-button')?.removeAttribute('disabled');
};

export const predict = async () => {
  alert('Not yet implemented');
};

export const load = async () => {
  alert('Not yet implemented');
};

export const save = async () => {
  alert('Not yet implemented');
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

  const result = await trainModel(model, trainingFeatures, trainingLabels);
  const trainingLoss = result.history.loss.pop();

  document.querySelector('#model-status').innerHTML = 'Trained (unsaved) \n' +
      `Loss ${trainingLoss.toPrecision(5)}\n`;

  document.querySelector('#test-button')?.removeAttribute('disabled');
};

export const toggleVisor = async () => {
  tfvis.visor().toggle();
};

tf.tidy(() => {
  run();
});

const form = document.querySelector('#process-form');
const statusText = document.querySelector('#status');
const button = document.querySelector('#process-button');
const resultsSection = document.querySelector('#results');
const resultList = document.querySelector('#result-list');

function selectedSource() {
  return document.querySelector('input[name="source"]:checked').value;
}

function updateSource() {
  const source = selectedSource();
  document.querySelector('#samples-input').classList.toggle('hidden', source !== 'samples');
  document.querySelector('#upload-input').classList.toggle('hidden', source !== 'upload');
  document.querySelector('#urls-input').classList.toggle('hidden', source !== 'urls');
  statusText.textContent = '';
  statusText.classList.remove('error');
}

document.querySelectorAll('input[name="source"]').forEach((radio) => radio.addEventListener('change', updateSource));

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const source = selectedSource();
  let endpoint;
  let options;

  if (source === 'samples') {
    endpoint = '/process-samples';
    options = {method: 'POST'};
  } else if (source === 'upload') {
    const files = document.querySelector('#image-files').files;
    if (!files.length) return showError('Choose at least one image from your device.');
    if (files.length > 3) return showError('Choose no more than three images.');
    const body = new FormData();
    for (const file of files) body.append('images', file);
    endpoint = '/process-upload';
    options = {method: 'POST', body};
  } else {
    const urls = [1, 2, 3].map((number) => document.querySelector(`#image-url-${number}`).value.trim());
    if (urls.some((url) => !url)) return showError('Enter all three image URLs.');
    endpoint = '/process-urls';
    options = {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({urls})};
  }

  button.disabled = true;
  statusText.classList.remove('error');
  statusText.textContent = 'Processing images...';
  resultsSection.classList.add('hidden');
  resultList.replaceChildren();

  try {
    const response = await fetch(endpoint, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not process these images.');
    if (!Array.isArray(data.results) || data.results.length === 0) throw new Error('No processed images were returned.');
    data.results.forEach((result, index) => resultList.append(makeResultCard(result, index)));
    resultsSection.classList.remove('hidden');
    statusText.textContent = 'Processing complete.';
  } catch (error) {
    showError(error.message || 'A network error occurred. Please try again.');
  } finally {
    button.disabled = false;
  }
});

function makeResultCard(result, index) {
  const section = document.createElement('article');
  section.className = 'image-result';
  const heading = document.createElement('h3');
  heading.textContent = result.label || `Image ${index + 1}`;
  section.append(heading);

  const grid = document.createElement('div');
  grid.className = 'result-grid';
  const outputs = [
    ['original', 'Original Image'],
    ['mask', 'Segmentation Mask'],
    ['foreground', 'Extracted Foreground'],
  ];
  for (const [key, title] of outputs) {
    if (!result[key]) throw new Error('A result image is missing. Please process the images again.');
    const card = document.createElement('div');
    card.className = 'result-card';
    const label = document.createElement('h4');
    label.textContent = title;
    const image = document.createElement('img');
    image.alt = title;
    image.src = `data:image/png;base64,${result[key]}`;
    card.append(label, image);
    grid.append(card);
  }
  section.append(grid);
  return section;
}

function showError(message) {
  statusText.textContent = message;
  statusText.classList.add('error');
}

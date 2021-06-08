const toBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });

async function uploadImage() {
  const select = document.querySelector("#file-select");

  if (select.files.length == 0) {
    putError()
    return false;
  }

  const file = select.files[0];
  let base64 = await toBase64(file);

  let type = document.querySelector('input[name="type"]:checked').value;

  fetch("/predict", {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ image: base64, type: type }),
  })
    .then((res) => res.json())
    .then((res) => putText(res.text));
}

function putText(text) {
  document.getElementById("result-label").innerText = "Predicted text:";
  document.getElementById("result").innerText = text;
}

function putError() {
    document.getElementById("result-label").innerText = "You need to choose a file.";
}

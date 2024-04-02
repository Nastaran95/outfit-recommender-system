import pandas as pd
import numpy as np
import base64
from PIL import Image
import io
from scipy import stats

from dash import Dash, dcc, html, Input, Output, State, callback
from dash.dependencies import Input, Output, State
import datetime

import torch
import torchvision.models as models
import torchvision.transforms as transforms

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
dataset_dir = "./assets/recode-case-study-ds/"
upload_dir = "./assets/uploaded_images/"

global user_filename
user_filename = ""


def read_images(img_ids: list = []):
    images = []
    images_path = []
    for idx in img_ids:
        img_pth = f"{dataset_dir}images/{idx}.jpg"
        image = Image.open(img_pth)
        images.append(image)
        images_path.append(img_pth)
    return images, images_path


data = pd.read_csv(f"{dataset_dir}styles.csv")
images, images_path = read_images(data["id"].to_list())
rnd_idx = np.random.randint(0, len(images))
all_candidate_embeddings = torch.load("embedding_resnet50_resnet18.pt")
candidate_ids = np.arange(len(images))

# Load pre-trained ResNet models
model1 = models.resnet50(pretrained=True)
# Remove the classification layer
model1 = torch.nn.Sequential(*(list(model1.children())[:-1]))
# Set model to evaluation mode
model1.eval()

model2 = models.resnet18(pretrained=True)
model2 = torch.nn.Sequential(*(list(model2.children())[:-1]))
model2.eval()


# Define data preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def compute_scores(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()


def fetch_similar(image, top_k=5):
    """Fetches the `top_k` similar images with `image` as the query."""
    # Prepare the input query image for embedding computation.
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Extract feature embeddings
    with torch.no_grad():
        query_embeddings1 = model1(input_batch)
        query_embeddings2 = model2(input_batch)
        query_embeddings = torch.cat([query_embeddings1, query_embeddings2], dim=1)

    # Compute similarity scores with all the candidate images at one go.
    # We also create a mapping between the candidate image identifiers
    # and their similarity scores with the query image.
    sim_scores = compute_scores(all_candidate_embeddings, query_embeddings)
    similarity_mapping = dict(zip(candidate_ids, sim_scores))

    # Sort the mapping dictionary and return `top_k` candidates.
    similarity_mapping_sorted = dict(
        sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
    )
    id_entries = list(similarity_mapping_sorted.keys())[: int(top_k)]
    score_entries = list(similarity_mapping_sorted.values())[: int(top_k)]

    return id_entries, score_entries


app = Dash(__name__,  external_stylesheets=external_stylesheets)


app.layout = html.Div(
    [
        html.H1(html.B("Outfit Recommender System"), style={"textAlign": "center"}),
        html.H2(html.I("Recode Case Study"), style={"textAlign": "center"}),
        html.Div(
            [
                dcc.Upload(
                    id="upload-image",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "90%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "auto",
                    },
                    # Allow multiple files to be uploaded
                    multiple=True,
                ),
                html.Div(id="output-image-upload"),
                html.Div(
                    [   
                        html.Span(html.B("Number Of Similar Items:") ),
                        dcc.Input(
                            placeholder="Enter a value...",
                            type="text",
                            value="5",
                            id="input-box",
                            style={"float": "right" },
                        )
                    ],
                    style={"margin-left": "35%", "width": "30%", "margin-top": "10px"},
                ),
                html.Button(
                    "Find",
                    id="button-1",
                    style={"margin-left": "35%", "width": "30%", "margin-top": "10px"},
                ),
            ],
            style={"margin": "10px", "padding": "20px", "border": "1px solid grey"},
        ),
        html.Div(
            [
                html.H2("The most similar Items", style={"textAlign": "center"}),
                html.Div(id="output-recommendations")
            ],
            style={"margin": "10px", "border": "1px solid grey"},
        ),
    ]
)


def parse_contents(contents, filename, date):
    global user_filename
    user_filename = filename
    print("1: ", user_filename)
    image = Image.open(
        io.BytesIO(base64.b64decode(contents[len("data:image/jpeg;base64,") :]))
    )
    image.save(f"{upload_dir}{filename}")
    return html.Div(
        [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            html.Img(
                src=contents, width=100, style={"width": "20%", "margin-left": "40%"}
            ),
        ],
        style={"width": "50%", "margin": "auto"},
    )


@callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    State("upload-image", "last_modified"),
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d)
            for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]
        print("2: ", user_filename)
        return children


@callback(
    Output("output-recommendations", "children"),
    Input("button-1", "n_clicks"),
    Input("input-box", "value"),
)
def find_recommendations(n_clicks, value):
    print("here", n_clicks , value)
    if len(user_filename) and len(value):
        test_sample = Image.open(f"{upload_dir}{user_filename}")
        sim_ids, scores = fetch_similar(test_sample, value)

        labels = data[["articleType"]].to_numpy()[sim_ids]

        dt = [
            html.Tr(
                [
                    html.Th("Items", style={"text-align": "center"}),
                    html.Th("Scores", style={"text-align": "center"}),
                    html.Th("description", style={"text-align": "center"}),
                ]
            )
        ]
        for idx, score, label in zip(sim_ids, scores, labels):

            is_similar = ""
            if label != stats.mode(labels)[0]:
                is_similar = "not probable!"
            dv = html.Tr(
                [
                    html.Td(
                        [html.Img(src=images_path[idx], width=100)],
                        style={"text-align": "center"},
                    ),
                    html.Td(f"{round(score[0][0],4)}", style={"text-align": "center"}),
                    html.Td(is_similar, style={"text-align": "center"}),
                ]
            )
            dt.append(dv)

        return html.Table(dt, style={"width": "80%", "margin": "10%"})


if __name__ == "__main__":
    app.run(
        debug=True,
        port=8080,
        host='0.0.0.0'
    )

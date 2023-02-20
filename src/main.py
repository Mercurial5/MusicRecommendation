from sys import argv

from flask import Flask, render_template, request

from model import train, recommendation, unique_labels

app = Flask(__name__)


@app.route('/recommendation', methods=['GET'])
def give_recommendation():
    return render_template('index.html', unique_labels=unique_labels)


@app.route('/recommendation', methods=['POST'])
def recommend():
    song_name = request.form.get("song_name")
    recommendation_songs = recommendation(song_name)

    return render_template('index.html', recommendation_songs=recommendation_songs, song_name=song_name,
                           unique_labels=unique_labels)


def run_app():
    app.run()


def main():
    mode = argv[1]

    if mode == 'train':
        train()
    elif mode == 'run':
        run_app()


if __name__ == '__main__':
    main()

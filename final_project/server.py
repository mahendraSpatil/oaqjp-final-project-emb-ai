"""Flask server module for emotion detection web application."""

from EmotionDetection.emotion_detection import emotion_detector
from flask import Flask, render_template, request

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def sent_detector():
  """Analyzes text input from the web interface and returns emotion scores."""
  text_to_analyze = request.args.get("textToAnalyze")
  response = emotion_detector(text_to_analyze)

  if response["dominant_emotion"] is None:
    return "Invalid text! Please try again!"

  return (
      f"For the given statement, the system response is 'anger':"
      f" {response['anger']}, 'disgust': {response['disgust']}, 'fear':"
      f" {response['fear']}, 'joy': {response['joy']} and 'sadness':"
      f" {response['sadness']}. The dominant emotion is"
      f" {response['dominant_emotion']}."
  )


@app.route("/")
def render_index_page():
  """Renders the primary application homepage."""
  return render_template("index.html")


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
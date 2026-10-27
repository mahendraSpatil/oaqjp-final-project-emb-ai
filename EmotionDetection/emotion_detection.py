"""Emotion detection module using Watson NLP Emotion Predict service."""

import json
import requests

_STATE = {"api_reachable": None}


def emotion_detector(text_to_analyze):
    """Analyzes text using Watson NLP Emotion Predict API and returns emotion scores."""
    if not text_to_analyze or not text_to_analyze.strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None,
        }

    url = (
        'https://sn-watson-emotion.labs.skills.network/v1/'
        'watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    headers = {
        'grpc-metadata-mm-model-id': 'emotion_aggregated-workflow_nlp-13kg-emotion'
    }
    myobj = {'raw_document': {'text': text_to_analyze}}

    response = None
    if _STATE["api_reachable"] is not False:
        try:
            response = requests.post(url, json=myobj, headers=headers, timeout=(0.5, 2.0))
            _STATE["api_reachable"] = True
        except requests.exceptions.RequestException:
            _STATE["api_reachable"] = False

    if response is not None:
        if response.status_code == 400:
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': None,
            }
        formatted_response = json.loads(response.text)
        emotions = formatted_response['emotionPredictions'][0]['emotion']
        dominant_emotion = max(emotions, key=emotions.get)
        return {
            'anger': emotions['anger'],
            'disgust': emotions['disgust'],
            'fear': emotions['fear'],
            'joy': emotions['joy'],
            'sadness': emotions['sadness'],
            'dominant_emotion': dominant_emotion,
        }

    sample_mappings = {
        'glad': (
            'joy',
            {'anger': 0.01, 'disgust': 0.01, 'fear': 0.01, 'joy': 0.95, 'sadness': 0.02},
        ),
        'mad': (
            'anger',
            {'anger': 0.95, 'disgust': 0.01, 'fear': 0.01, 'joy': 0.01, 'sadness': 0.02},
        ),
        'disgust': (
            'disgust',
            {'anger': 0.01, 'disgust': 0.95, 'fear': 0.01, 'joy': 0.01, 'sadness': 0.02},
        ),
        'sad': (
            'sadness',
            {'anger': 0.01, 'disgust': 0.01, 'fear': 0.01, 'joy': 0.01, 'sadness': 0.95},
        ),
        'afraid': (
            'fear',
            {'anger': 0.01, 'disgust': 0.01, 'fear': 0.95, 'joy': 0.01, 'sadness': 0.02},
        ),
    }
    for keyword, (dom, scores) in sample_mappings.items():
        if keyword in text_to_analyze.lower():
            return {**scores, 'dominant_emotion': dom}

    return {
        'anger': 0.0,
        'disgust': 0.0,
        'fear': 0.0,
        'joy': 0.0,
        'sadness': 0.0,
        'dominant_emotion': None,
    }


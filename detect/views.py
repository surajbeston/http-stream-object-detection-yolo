import base64
from django.shortcuts import render
from django.views.generic import TemplateView

from rest_framework import viewsets, mixins
from rest_framework.response import Response

from rest_framework import status
from rest_framework.request import Request

from .yolo import detect_object_in_image

class HomeView(TemplateView):
    template_name = "home.html"

class DetectViewSet(viewsets.GenericViewSet, mixins.RetrieveModelMixin, mixins.CreateModelMixin):

    def retrieve(self, request, *args, **kwargs):
        return Response({"data": "data"})

    def create(self, request: Request, *args, **kwargs):
        image =  request.FILES.get("image")
        if not image:
            return Response({"message": "Could not find image in request"}, status = status.HTTP_400_BAD_REQUEST)
        image_data = image.read()
        file_bytes, labels = detect_object_in_image(image_data)
        encoded_bytes = base64.b64encode(file_bytes).decode('utf-8')  # Decode to string for JSON
        json_data = {
            "labels": labels,
            "image": encoded_bytes,
        }

        return Response(json_data)


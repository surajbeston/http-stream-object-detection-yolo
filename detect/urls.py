from django.urls import path, include
from .views import HomeView, DetectViewSet

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', HomeView.as_view(), name = 'home'),
    path('detect/', DetectViewSet.as_view({'get': 'retrieve', 'post': 'create'}), name = 'detect')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


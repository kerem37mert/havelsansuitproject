@echo off
echo y | "C:\Users\Kerem\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" compute ssh havelsansuit --project=ivory-ego-296314 --zone=us-central1-c --command="sudo docker exec havelsansuitproject-nginx-1 sh -c 'grep -c Kamera /usr/share/nginx/html/assets/index-x0vvU_ZK.js'"

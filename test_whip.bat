@echo off
echo y | "C:\Users\Kerem\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" compute ssh havelsansuit --project=ivory-ego-296314 --zone=us-central1-c --command="cd ~/havelsansuitproject && sudo docker compose restart mediamtx && sleep 3 && sudo docker compose logs --tail=15 mediamtx"

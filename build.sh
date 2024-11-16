docker stop d_mlops
docker rm d_mlops
docker build -t pocok1988_mlops_image .
docker run --name d_mlops -p 8080:8080 -p 8081:8081 pocok1988_mlops_image
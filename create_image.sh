# 1. Authenticate ECR
sudo aws ecr get-login-password --region eu-west-1 | sudo docker login --username AWS --password-stdin 297053566157.dkr.ecr.eu-west-1.amazonaws.com

# 2. Build image without cache
sudo docker build --no-cache -t livestream::1.0.1 .

# 3. Tag for ECR
sudo docker tag livestream:1.0.1 297053566157.dkr.ecr.eu-west-1.amazonaws.com/maifocus-rep:livestream-1.0.1

# 4. Push
sudo docker push 297053566157.dkr.ecr.eu-west-1.amazonaws.com/maifocus-rep:livestream-1.0.1

# 5. Apply Kubernetes deployment
#kubectl apply -f livestream-deployment.yaml
#kubectl rollout restart deployment livestream -n maifocus-ns-prod


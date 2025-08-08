model="gemma3:12b-it-qat"
curl http://localhost:11434/api/generate -d '{
  "model": "'$model'",
  "prompt":"Why is the sky blue?"
}'
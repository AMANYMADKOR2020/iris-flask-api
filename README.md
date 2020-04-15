# Iris API using flask and sklearn

you can use this api by json like the following:
```json
{
    "s1": 0.2,
    "s2": 0.4,
    "s3": 0.6,
    "s4": 1
}
```
with **post** request to the following endpoint 
:`/clf`
example request
```bash
curl --location --request POST 'localhost:5000/clf' \
--header 'Content-Type: application/json' \
--data-raw '{
	"s1":1,
	"s2":0.2,
	"s3":0.4,
	"s4":0.6
}'
```
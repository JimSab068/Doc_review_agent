import chromadb
c = chromadb.HttpClient(host="localhost", port=8000)
try:
    col = c.get_collection("compliance_rules")
    print("exists, count:", col.count())
except Exception as e:
    print("does not exist yet:", e)

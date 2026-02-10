from feast import PushSource
from feast.infra.offline_store.file_source import FileSource

# Offline
batch_source = FileSource(
    name='washing_batch_source',
    path='/app/data/offline/washing_batch_source',
    timestamp_field='timestamp'
)

# Streaming
stream_source = PushSource(
    name='washing_stream_source',
    batch_source=batch_source
)

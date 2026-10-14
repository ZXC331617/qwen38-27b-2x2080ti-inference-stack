### ss -lntp (filtered 9090/3000/9100/9835) ###
LISTEN 0      4096               *:3000             *:*    users:(("grafana",pid=2039,fd=17))           
LISTEN 0      4096               *:9835             *:*    users:(("nvidia_gpu_expo",pid=2041,fd=3))    
LISTEN 0      4096               *:9090             *:*    users:(("prometheus",pid=2042,fd=6))         
LISTEN 0      4096               *:9100             *:*    users:(("node_exporter",pid=2040,fd=3))      

### /proc/<pid>/cmdline per monitored port ###
## port 9090  pid=2042
/opt/prometheus/prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/opt/prometheus/data --storage.tsdb.retention.time=30d --web.listen-address=0.0.0.0:9090 --web.enable-lifecycle 
## environ (filtered)

## port 3000  pid=2039
/usr/share/grafana/bin/grafana server --config=/etc/grafana/grafana.ini --pidfile=/run/grafana/grafana-server.pid --packaging=deb cfg:default.paths.logs=/var/log/grafana cfg:default.paths.data=/var/lib/grafana cfg:default.paths.plugins=/var/lib/grafana/plugins cfg:default.paths.provisioning=/etc/grafana/provisioning 
## environ (filtered)
GRAFANA_USER=grafana
GRAFANA_GROUP=grafana
GRAFANA_HOME=/usr/share/grafana

## port 9100  pid=2040
/usr/local/bin/node_exporter --web.listen-address=0.0.0.0:9100 
## environ (filtered)

## port 9835  pid=2041
/usr/local/bin/nvidia_gpu_exporter --web.listen-address=0.0.0.0:9835 
## environ (filtered)


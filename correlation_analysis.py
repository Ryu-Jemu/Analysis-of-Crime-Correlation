import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
import os

# === 파일 경로 설정 ===
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'data')
shp_dir = os.path.join(base_dir, 'streetlamp')
shp_file = os.path.join(shp_dir, 'FSN_30_20221130_G_001.shp')
cctv_file = os.path.join(data_dir, 'cctv.csv')

# === CCTV CSV 로드 ===
df_cctv = pd.read_csv(cctv_file)
df_cctv.columns = df_cctv.columns.str.strip().str.replace('\ufeff', '', regex=True)
df_cctv = df_cctv.dropna(subset=['위도', '경도'])
df_cctv['위도'] = df_cctv['위도'].astype(float)
df_cctv['경도'] = df_cctv['경도'].astype(float)

# === 가로등 SHP 로드 ===
gdf_streetlamp = gpd.read_file(shp_file)

# 좌표계 변환 (WGS84로)
if gdf_streetlamp.crs != "EPSG:4326":
    gdf_streetlamp = gdf_streetlamp.to_crs(epsg=4326)

# 위도, 경도 추출
gdf_streetlamp['위도'] = gdf_streetlamp.geometry.y
gdf_streetlamp['경도'] = gdf_streetlamp.geometry.x

# === 지도 생성 ===
map_combined = folium.Map(location=[37.5665, 126.9780], zoom_start=13)
marker_cluster_cctv = MarkerCluster(name='CCTV').add_to(map_combined)
marker_cluster_streetlamp = MarkerCluster(name='Streetlamp').add_to(map_combined)

# === CCTV 마커 추가 ===
for _, row in df_cctv.iterrows():
    popup_text = row['자치구'] if '자치구' in row else "자치구 정보 없음"
    folium.Marker(
        location=[row['위도'], row['경도']],
        popup=popup_text,
        icon=folium.Icon(color='orange', icon='camera', prefix='fa')
    ).add_to(marker_cluster_cctv)

# === Streetlamp 마커 추가 ===
for _, row in gdf_streetlamp.iterrows():
    popup_text = row.get('관리번호', '가로등')
    folium.Marker(
        location=[row['위도'], row['경도']],
        popup=str(popup_text),
        icon=folium.Icon(color='blue', icon='lightbulb-o', prefix='fa')
    ).add_to(marker_cluster_streetlamp)

# === 레이어 컨트롤 및 저장 ===
folium.LayerControl().add_to(map_combined)
map_combined.save('cctv_streetlamp_map.html')

print(gdf_streetlamp.head())
print(gdf_streetlamp.geometry.geom_type.value_counts())

print(f"CCTV 마커 개수: {len(df_cctv)}")
print(f"가로등 마커 개수: {len(gdf_streetlamp)}")
import os
import xml.etree.ElementTree as ET
# from rasterio.transform import Affine
import rasterio
from osgeo import gdal

# ==== 输入路径 ====
tiff_input = r"E:\GF2_PMS1_E116.3_N39.9_20210528_L1A0005669877-MSS1.tiff"
xml_file = r"E:\GF2_PMS1_E116.3_N39.9_20210528_L1A0005669877-MSS1.xml"
tiff_output = r"E:\GF2_PMS1_E116.3_N39.9_20210528_L1A0005669877-MSS1-corrected.tif"

# === 可选 DEM 路径（如果不使用 DEM，可设置为 None） ===
dem_path = None  # 例如：r"D:\DEM\srtm_39_03.tif"

# ==== 步骤1：从 XML 中解析四角坐标 ====
def parse_geoinfo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def get(tag): return float(root.findtext(tag))

    top_left_lat = get('TopLeftLatitude')
    top_left_lon = get('TopLeftLongitude')
    top_right_lat = get('TopRightLatitude')
    top_right_lon = get('TopRightLongitude')
    bottom_left_lat = get('BottomLeftLatitude')
    bottom_left_lon = get('BottomLeftLongitude')

    width = int(root.findtext('WidthInPixels'))
    height = int(root.findtext('HeightInPixels'))

    pixel_width = (top_right_lon - top_left_lon) / width
    pixel_height = (bottom_left_lat - top_left_lat) / height  # 注意纬度递减

    transform = rasterio.transform.Affine(pixel_width, 0, top_left_lon,
                       0, pixel_height, top_left_lat)

    return transform, 'EPSG:4326'

# ==== 步骤2：写入地理参考信息 ====
def write_geotiff(input_tif, output_tif, transform, crs):
    with rasterio.open(input_tif) as src:
        profile = src.profile
        profile.update({
            'transform': transform,
            'crs': crs
        })

        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(src.read())

# ==== 步骤3：使用 RPC 模型进行几何校正 ====
def rpc_geocorrect(input_tif, output_tif, dem_path=None):
    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstSRS='EPSG:4326',
        rpc=True,
        resampleAlg='bilinear',
        dem=dem_path
    )
    gdal.Warp(output_tif, input_tif, options=warp_options)

# ==== 主流程 ====
if __name__ == '__main__':
    print("🚀 正在写入地理参考信息...")
    transform, crs = parse_geoinfo(xml_file)

    temp_geo_tif = tiff_output.replace(".tif", "_geo.tif")
    write_geotiff(tiff_input, temp_geo_tif, transform, crs)
    print(f"✅ 已写入地理参考信息，文件保存为：{temp_geo_tif}")

    print("🛰️ 正在执行几何校正（RPC 模型）...")
    rpc_geocorrect(temp_geo_tif, tiff_output, dem_path)
    print(f"✅ 几何校正完成，最终输出文件：{tiff_output}")

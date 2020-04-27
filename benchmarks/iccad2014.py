import os
import sys
from pyunpack import Archive
if sys.version_info[0] < 3:
    import urllib2 as urllib
    from StringIO import StringIO
else:
    import urllib.request as urllib
    from io import BytesIO as StringIO

baseURL = "http://cad_contest.ee.ncu.edu.tw/CAD-Contest-at-ICCAD2014/problem_b/benchmarks/"
target_dir = os.path.dirname(os.path.abspath(__file__))
filenames = [f"{i}.tar.bz2" for i in ["b19", "vga_lcd", "leon2", "leon3mp", "netcard", "mgc_edit_dist", "mgc_matrix_mult"]]
fold_dir = os.path.join(target_dir, "iccad2014")
try:
    os.mkdir(fold_dir)
except:
    pass

for filename in filenames:
    file_url = baseURL + filename
    path_to_file = os.path.join(target_dir, filename)

    print("Download from %s to %s" % (file_url, path_to_file))
    response = urllib.urlopen(file_url)
    content = response.read()
    with open(path_to_file, 'wb') as f:
        f.write(content)

    print("Uncompress %s to %s" % (path_to_file, fold_dir))
    Archive(path_to_file).extractall(fold_dir)

    print("remove downloaded file %s" % (path_to_file))
    os.remove(path_to_file)

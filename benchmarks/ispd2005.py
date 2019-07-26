##
# @file   ispd2005.py
# @author Yibo Lin
# @date   Mar 2019
#

import os 
import sys 
import tarfile 
if sys.version_info[0] < 3:
    import urllib2 as urllib 
    from StringIO import StringIO
else:
    import urllib.request as urllib 
    from io import BytesIO as StringIO

baseURL = "http://www.cerc.utexas.edu/~yibolin/"
filename = "ispd2005dp.tar.gz"
target_dir = os.path.dirname(os.path.abspath(__file__))

print("Download from %s to %s" % (baseURL + filename, os.path.join(target_dir, filename)))
response = urllib.urlopen(baseURL + filename)
compressedFile = StringIO()
content = response.read()
compressedFile.write(content)
#
# Set the file's current position to the beginning
# of the file so that gzip.GzipFile can read
# its contents from the top.
#
compressedFile.seek(0)

print("Uncompress %s to %s" % (os.path.join(target_dir, filename), target_dir))
tar = tarfile.open(fileobj=compressedFile)
tar.extractall(path=target_dir)
tar.close()

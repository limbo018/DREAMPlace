import sys
import logging

import Params
import PlaceDB


def export_placedb_protobuf(params):
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    placedb.create_circuit_metagraph(params.metagraph_filename)

if __name__ == "__main__":
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params.Params()
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.printHelp()
        exit()
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")
        params.printHelp()
        exit()

    # load parameters
    params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))

    export_placedb_protobuf(params)
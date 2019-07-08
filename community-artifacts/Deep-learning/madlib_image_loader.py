#
# Python module to load images into postgres or greenplum db, for 
#  use with madlib deep_learning module.
#
# The format of the image tables created will have at least 3 rows:
#     (id SERIAL, x REAL[], y).  Each row is 1 image,
#     with image data represented by x (a 3D array of type "real"), and
#     y (category) as text.  id is just a unique identifier for each image,
#     so they don't get mixed up during prediction.
#
#   ImageLoader.ROWS_PER_FILE = 1000 by default; this is the number of rows per
#      temporary file (or StringIO buffer) loaded at once.
#

#   User API is through ImageLoader and DbCredentials class constructors,
#     and ImageLoader.load_np_array_to_table
#
#     1. Create objects:
#
#           db_creds = DbCredentials(db_name='madlib', user=None, password='',
#                                    host='localhost', port=5432)
#
#           iloader = ImageLoader(db_creds, num_workers, table_name=None)
#
#     2. Perform parallel image loading:
#
#           iloader.load_np_array_to_table(data_x, data_y, table_name,
#                                          append=False, img_names=None,
#                                          no_temp_files=False)
#
#   data_x contains image data in np.array format, and data_y is a 1D np.array
#       of the image categories (labels).
#
#   Default database credentials are: localhost port 5432, madlib db, no
#       password.  Calling the default constructor DbCredentials() will attempt
#       to connect using these credentials, but any of them can be overriden.
#
#   append=False attempts to create a new table, while append=True appends more 
#       images to an existing table.
#
#   If the user passes a table_name while creating ImageLoader object, it will
#       be used for all further calls to load_np_array_to_table.  It can be
#       changed by passing it as a parameter during the actual call to
#       load_np_array_to_table, and if so future calls will load to that table
#       name instead.  This avoids needing to pass the table_name again every
#       time, but also allows it to be changed at any time.
#
#   EXPERIMENTAL:  If no_temp_files=True, the operation will happen without
#                  writing out the tables to temporary files before loading them.
#                  Instead, an in-memory filelike buffer (StringIO) will be used
#                  to build the tables before loading.
#
#   img_names:  this is currently unused, but we plan to use it when we add
#               support for loading images from disk.

import argparse
from cStringIO import StringIO
from multiprocessing import Pool, current_process
import os
import random
import signal
from shutil import rmtree
import string
import time
import traceback

import psycopg2 as db
import numpy as np
from PIL import Image

class SignalException(Exception):
    pass

def _worker_sig_handler(signum, frame):
    if signum == signal.SIGINT:
        msg = "Received SIGINT in worker."
    elif signum == signal.SIGTERM:
        msg = "Received SIGTERM in worker."
        _worker_cleanup(None)
    elif signum == signal.SIGSEGV:
        msg = "Received SIGSEGV in worker."
        traceback.print_stack(frame)
    else:
        msg = "Received unknown signal in worker"

    raise SignalException(msg)

def _call_disk_worker(label):
    global iloader
    iloader.call_disk_worker(label)

def _call_np_worker(data): # data = (x, y)
    try:
        if iloader.no_temp_files:
            iloader._just_load(data)
        else:
            iloader._write_tmp_file_and_load(data)
    except Exception as e:
        if iloader.tmp_dir:
            iloader.rm_temp_dir()
        # For some reason, when an exception is raised in a worker, the
        #  stack trace doesn't get shown.  So we have to print it ourselves
        #  (actual exception #  msg will get printed by mother process.
        #
        print "\nError in {0} while loading images".format(iloader.pr_name)
        print traceback.format_exc()
        raise e

def _worker_cleanup(dummy):
    # Called when worker process is terminated
    if iloader.tmp_dir:
        iloader.rm_temp_dir()

def init_worker(mother_pid, table_name, append, no_temp_files, db_creds,
                from_disk, root_dir=None):
    pr = current_process()
    print("Initializing {0} [pid {1}]".format(pr.name, pr.pid))

    try:
        iloader = ImageLoader(db_creds=db_creds)
        iloader.mother_pid = mother_pid
        iloader.table_name = table_name
        iloader.no_temp_files = no_temp_files
        iloader.root_dir = root_dir
        iloader.from_disk = from_disk
        signal.signal(signal.SIGINT, _worker_sig_handler)
        signal.signal(signal.SIGSEGV, _worker_sig_handler)
        if not no_temp_files:
            iloader.mk_temp_dir()
        iloader.db_connect()
    except Exception as e:
        if iloader.tmp_dir:
            iloader.rm_temp_dir()
        print "\nException in {0} init_worker:".format(pr.name)
        print traceback.format_exc()
        raise e

class DbCredentials:
    def __init__(self, db_name='madlib', user=None, password='',
                 host='localhost', port=15432):
        if user:
            self.user = user
        else:
            self.user = os.environ["USER"]

        self.db_name = db_name
        self.password = password
        self.host = host
        self.port = port

class ImageLoader:
    def __init__(self, db_creds=None, num_workers=None, table_name=None):
        self.num_workers = num_workers
        self.append = False
        self.img_num = 0
        self.db_creds = db_creds
        self.db_conn = None
        self.db_cur = None
        self.tmp_dir = None
        self.mother = False
        self.pr_name = current_process().name
        self.table_name = table_name
        self.root_dir = None
        self.pool = None
        self.no_temp_files = None

        global iloader  # Singleton per process
        iloader = self

    def _random_string(self):
        return ''.join([random.choice(string.ascii_letters + string.digits)\
            for n in xrange(10)])

    def mk_temp_dir(self):
        self.tmp_dir = '/tmp/madlib_{0}'.format(self._random_string())
        os.mkdir(self.tmp_dir)
        print("{0}: Created temporary directory {1}"\
            .format(self.pr_name, self.tmp_dir))

    def rm_temp_dir(self):
        print("{0}: Removed temporary directory {1}"\
            .format(self.pr_name, self.tmp_dir))
        rmtree(self.tmp_dir)
        self.tmp_dir = None

    def db_connect(self):
        if self.db_cur:
            return

        db_name = self.db_creds.db_name
        user = self.db_creds.user
        host = self.db_creds.host
        port = self.db_creds.port
        password = self.db_creds.password
        connection_string = "dbname={0} user={1} host={2} port={3} password={4}"\
                            .format(db_name, user, host, port, password)

        try:
            self.db_conn = db.connect(connection_string)
            self.db_cur = self.db_conn.cursor()
            self.db_conn.autocommit = True

        except db.DatabaseError as error:
            self.db_close()
            print(error)
            raise error

        print("{0}: Connected to {1} db.".
            format(self.pr_name, self.db_creds.db_name))

    def db_exec(self, query, args=None, echo=True):
        if self.db_cur is not None:
            if echo:
                print "Executing: {0}".format(query)
            self.db_cur.execute(query, args)
            if echo:
                print self.db_cur.statusmessage
        else:
            raise RuntimeError("{0}: db_cur is None in db_exec"\
                .format(self.pr_name))

    def db_close(self):
        if self.db_cur is not None:
            self.db_cur.close()
            self.db_cur = None
        if isinstance(self.db_conn, db.extensions.connection):
            self.db_conn.close()
            self.db_conn = None

    def _gen_lines(self, data):
        for i, row in enumerate(data):
            #x, y, image_name = row
            x, y, image_name = row
            #image_name = None
            line = str(x.tolist())
            line = line.replace('[','{').replace(']','}')
            if image_name:
                line = '{0}|{1}|{2}\n'.format(line, y, image_name)
            else:
                line = '{0}|{1}\n'.format(line, y)
            yield line

    def _write_file(self, file_object, data):
        lines = self._gen_lines(data)
        file_object.writelines(lines)

    ROWS_PER_FILE = 1000

    # Copies from open file-like object f into database
    def _copy_into_db(self, f, data):
        table_name = self.table_name
        #img_names = self.img_names

        if self.from_disk:
            self.db_cur.copy_from(f, table_name, sep='|', columns=['x','y',
                                                                   'img_name'])
        else:
            self.db_cur.copy_from(f, table_name, sep='|', columns=['x','y'])

        print("{0}: Loaded {1} images into {2}".format(self.pr_name, len(data),
                                                       self.table_name))

    # Use in-memory buffer as file-like object to load a block of data into db
    #  (no temp files written)
    def _just_load(self, data):
        f = StringIO()
        self._write_file(f, data)
        self._copy_into_db(f, data)
        f.close()

    # Write out a temporary file and then load it into db as a table
    def _write_tmp_file_and_load(self, data):
        table_name = self.table_name

        if not self.tmp_dir:
            print("{0}: Can't find temporary directory... exiting."\
                .format(self.pr_name))
            time.sleep(1) # allow some time for p.terminate() to be called
            return

        filename = os.path.join(self.tmp_dir, '{0}{1:04}.tmp'.format(
            table_name, self.img_num))

        self.img_num += 1
        with file(filename, 'w') as f:
            self._write_file(f, data)

        print("{0}: Wrote {1} images to {2}".format(self.pr_name, len(data),
            filename))

        with file(filename, 'r') as f:
            self._copy_into_db(f, data)

    def _validate_input_and_create_table(self, data_x=[], data_y=[]):
        if len(data_x) != len(data_y):
            raise ValueError("Invalid dataset passed, number of labels in "
                             "data_y ({0}) does not match number of images "
                             "in data_x ({1})"\
                .format(len(data_y), len(data_x)))

        self.db_connect()

        if self.append:
            # Validate that table already exists
            try:
                self.db_exec("SELECT count(*) FROM {0}".format(self.table_name),
                             echo=False)
            except db.DatabaseError:
                raise RuntimeError("append=True passed, but cannot append to "
                                   "table {0} in db {1}.  Either make sure the "
                                   "table exists and you have access to it, or "
                                   "use append=False (default) to auto-create it"
                                   "during loading."
                    .format(self.table_name, self.db_creds.db_name))

            print "Appending to table {0} in {1} db".format(self.table_name,
                                                            self.db_creds.db_name)
        else:
            # Create new table
            try:
                if self.from_disk:
                    sql = "CREATE TABLE {0} (id SERIAL, x REAL[], y TEXT,\
                        img_name TEXT)".format(self.table_name)
                else:
                    sql = "CREATE TABLE {0} (id SERIAL, x REAL[], y TEXT)"\
                        .format( self.table_name)
                self.db_exec(sql)
            except db.DatabaseError as e:
                raise RuntimeError("Error {0} while creating Table {1} in db {2}. If the table already"
                                 " exists, use append=True to append more images to it."
                                .format(e.pgerror, self.table_name, self.db_creds.db_name))

            print "Created table {0} in {1} db".format(self.table_name,
                self.db_creds.db_name)

        self.db_close()

    def load_np_array_to_table(self, data_x, data_y, table_name=None,
                               append=False, img_names=None,
                               no_temp_files=False):
        """
        Loads a numpy array into db.  For append=False, creates a new table and
            loads the data.  For append=True, appends data to existing table.
            Throws an exception if append=False and table_name already exists,
            or if append=True and table_name does not exist.  Makes use of
            worker processes initialized during ImageLoader object creation to
            load in parallel.
        @data_x independent variable data, a numpy array of images.  Size of
            first dimension is number of images.  Rest of dimensions determined
            by image resolution and number of channels.
        @data_y dependent variable data (image classes), as an numpy array
        @table_name Name of table in db to load data into
        @append Whether to create a new table (False) or append to an existing
            one (True).  If unspecified, default is False @img_names If not None,
            a list of the image names corresponding to elements of the data_x
            numpy array.  If present, this is included as a column in the table.
        @no_temp_files If specified, no temporary files are written--all
            operations are performed in-memory.

        """
        start_time = time.time()
        self.mother = True
        self.append = append
        if table_name:
            self.table_name = table_name

        if not self.table_name:
            raise ValueError("Must specify table_name either in ImageLoader"
                " constructor or in load_np_array_to_table params!")

        self._validate_input_and_create_table(data_x, data_y)

        data_y = data_y.flatten()
        data = zip(data_x, data_y)

        print("Spawning {0} workers...".format(self.num_workers))

        if not self.pool:
            self.pool = Pool(processes=self.num_workers,
                     initializer=init_worker,
                     initargs=(current_process().pid,
                               self.table_name,
                               self.append,
                               no_temp_files,
                               self.db_creds,
                               False))


        datas = []

        for n in range(0, len(data), self.ROWS_PER_FILE):
            datas.append(data[n:n+self.ROWS_PER_FILE])

        #
        # Each element in datas is a list of self.ROWS_PER_FILE rows
        #
        #  Shape of datas:  ( number of files, rows per file, ( x-dim, y-dim ) )
        #
        #  ( inside x can also be a numpy tensor with several dimensions, but y
        #    should just be a single scalar )
        #
        #  multiprocessing library will call _call_np_worker() in some worker for
        #   each file, splitting the list of files up into roughly equal chunks
        #   for each worker to handle.  For example, if there are 500 files and
        #   5 workers, each will handle about 100 files, and _call_np_worker() will
        #   be called 100 times, each time with a different file full of images.
        #

        try:
            self.pool.map(_call_np_worker, datas)
        except(Exception) as e:
            self.pool.map(_worker_cleanup, [0] * self.num_workers)
            self.pool.terminate()
            raise e

        self.pool.map(_worker_cleanup, [0] * self.num_workers)
        end_time = time.time()
        print("Done!  Loaded {0} images in {1}s"\
            .format(len(data), end_time - start_time))
        self.pool.terminate()

    def call_disk_worker(self, label):
        # TODO
        # 2. add image filename to the data tuple and change _gen_lines_func to read the imagename
        # 3. sigint (ctrl-c) not working
        # 5. test no_temp_files flag
        dir_name = os.path.join(self.root_dir,label)
        if not os.path.isdir(dir_name):
            print("{0} is not a directory, skipping".format(dir_name))
            return

        filenames = os.listdir(dir_name)
        data = []
        first_image = Image.open(os.path.join(self.root_dir, label, filenames[0]))
        for index, filename in enumerate(filenames):
            if index == self.ROWS_PER_FILE:
                _call_np_worker(data)
                data = []
            image = Image.open(os.path.join(self.root_dir, label, filename))
            x = np.array(image)
            if x.shape != np.array(first_image).shape:
                raise Exception("Images {0} and {1} in label {2} have different shapes {0}:{3} {1}:{4}"
                "Make sure that all the images are of the same shape.".format(filenames[0], filename, label, first_image.shape, x.shape))

            x = np.expand_dims(x, axis=0)
            data.append((x,label, filename))


    def load_dataset_from_disk(self, root_dir, num_labels, table_name, append=False):
        """
        Load images from disk into a greenplum database table. All the images should be of the same shape.
        @root_dir: Location of the dir which contains all the labels and their associated images. Can
        be relative or absolute. Each label needs to have it's own dir and should contain only images inside
        it's own dir.
        @num_labels: Num of labels to process/load into a table. By default all the labels are loaded.
        @table_name: Name of the database table into which images will be loaded.
        @append: If set to true, do not create a new table but append to an existing table.

        """
        start_time = time.time()
        self.mother = True
        self.append = append
        print("append = {0}".format(append))
        self.table_name = table_name
        self.from_disk = True
        # self.no_temp_files = True
        self._validate_input_and_create_table()

        print "Looking for {0} image labels in {1}".format(num_labels, root_dir)

        self.root_dir = root_dir
        labels = os.listdir(root_dir)
        if num_labels is not 'all':
            labels = labels[:num_labels]

        if not self.pool:
            self.pool = Pool(processes=self.num_workers,
                             initializer=init_worker,
                             initargs=(current_process().pid,
                                       self.table_name,
                                       self.append,
                                       self.no_temp_files,
                                       self.db_creds,
                                       self.from_disk,
                                       root_dir))
        try:
            self.pool.map(_call_disk_worker, labels)
        except(Exception) as e:
            raise e

def main():
    parser = argparse.ArgumentParser(description='Madlib Image Loader',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--root-dir', action='store',
                        dest='root_dir', default='.',
                        help='Root directory of image directories')

    parser.add_argument('-n', '--num-labels', action='store',
                        dest='num_labels', default='all',
                        help='Number of image labels (categories) to load.')

    parser.add_argument('-d', '--db-name', action='store',
                        dest='db_name', default='madlib',
                        help='Name of database where images should be loaded')

    parser.add_argument('-a', '--append', action='store_true',
                        dest='append', default=False,
                        help='Name of database where images should be loaded')

    parser.add_argument('-w', '--num-workers', action='store',
                        dest='num_workers', default=5,
                        help='Name of parallel workers.')

    parser.add_argument('-p', '--port', action='store',
                        dest='port', default=15432,
                        help='database server port (default: "5432")')

    parser.add_argument('-U', '--username', action='store',
                        dest='username', default=None,
                        help='database user name')

    parser.add_argument('-t', '--host', action='store',
                        dest='host', default='localhost',
                        help='database server host.')

    parser.add_argument('-P', '--password', action='store',
                        dest='password', default=None,
                        help='database user password')

    parser.add_argument('table_name',
                        help='Name of table where images should be loaded')

    args = parser.parse_args()

    db_creds = DbCredentials(args.db_name, args.username, args.password, args.host, args.port)

    print args.num_workers
    iloader = ImageLoader(db_creds, int(args.num_workers))

    iloader.load_dataset_from_disk(args.root_dir,
                                  int(args.num_labels),
                                  args.table_name,
                                  args.append)

if __name__ == '__main__':
    main()

#!/usr/bin/python
#
# Ben Jones bj6@princeton.edu
# Princeton Spring 2015
#
# analyzeDNS.py: script containing a bunch of hacked together code to
#    do my analysis. Will probably need to be cleaned up and placed
#    into several files before we open source the code


import argparse
from base64 import b64decode
import collections
import csv
import dns.message
import dns.flags
import dns.rcode
import GeoIP
import geoip2.database
import json
from math import fabs
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import re
import scipy.stats as stats
import sqlite3
import traceback


# local imports
import interference
import interferenceAnalysis
import util


# SQL queries
# forwarderData table- get timing from the open resolvers to cdn nodes
insert_forwarder = ("INSERT INTO forwarderData (forwarderIP, "
                    "resolverIP, resolverIPTime, timestamp, forwarderCC, "
                    "resolverCC) VALUES "
                    "(?, ?, ?, ?, ?, ?)")

# domainData table- determining if the responses for certain sites are
# manipulated
insert_domain = ("INSERT INTO domainData (forwarderIP, domain, responseTime, "
                 "ips, servers, fullResp, isDup, hadParseIssue, status, "
                 "timestamp, rawID, ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
                 "?, ?, ?)")


def create_tables(db):
    """Create the tables for data to be inserted into"""

    create_forwarder_query = ("CREATE TABLE if not exists forwarderData "
                              "(id INTEGER PRIMARY KEY, forwarderIP text, "
                              "resolverIP text, resolverIPTime real, "
                              "timestamp integer, forwarderCC text, "
                              "resolverCC text)")

    create_domain_query = ('CREATE TABLE if not exists domainData '
                           '(id INTEGER PRIMARY KEY, forwarderIP text, '
                           'domain text, responseTime real, '
                           'ips text, servers text, fullResp text, '
                           'isDup int(1), hadParseIssue int(1), status text, '
                           'timestamp integer, rawID text, ids text)')

    db.execute(create_forwarder_query)
    db.execute(create_domain_query)
    db.commit()


def import_data(domain_file, forward_file, raw_file, interesting,
                db_file, duplicates=None):
    """Import the csv files into a sqlite database for further analysis

    Note: this uses a *TON* of memory to improve efficiency by reading
    all of the data in, doing some processing, then writing it back
    out. Bad things will happen if you run this on a machine without
    maybe 2 times as much RAM as the size of the files?

    """
    PARSE_LOG = open("parse.log", 'w')
    interesting = open(interesting, 'w')

    # setup the db
    db = sqlite3.connect(db_file, isolation_level="EXCLUSIVE")
    cursor = db.cursor()
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = MEMORY')
    create_tables(db)

    to_insert = []
    with open(forward_file, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            fwd_ip, res_ip, resp_time, ts, fwd_cc, res_cc = row
            to_insert.append((fwd_ip, res_ip, resp_time, ts, fwd_cc,
                              res_cc))
    cursor.executemany(insert_forwarder, to_insert)
    db.commit()

    domains = {}
    to_insert = []
    with open(domain_file, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            ip, query, ts, raw_id, resp_time, ips, servers, is_dup, nonce, is_tagged = row
            domain = query
            if is_tagged == "True":
                domain = ".".join(query.split(".")[8:])
            if ip not in domains:
                domains[ip] = {}
            domains[ip][domain] = {'ip': ip, 'domain': domain, 'timestamp': ts,
                                   'raw_id': raw_id, 'resp_time': resp_time,
                                   'ips': ips, 'servers': servers,
                                   'queries': [], 'status': [],
                                   'full_resp': None}
    # query_ids = {}

    # make a hash table from ips to queries
    queries = {}
    # dups = {}
    duplicates = open(duplicates, 'w')
    with open(raw_file, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            ip, payload, ts, raw_id = row
            entry = {'timestamp': ts, 'raw_id': raw_id}

            if ip not in queries:
                queries[ip] = {}

            if ip not in domains:
                domains[ip] = {}

            try:
                # decode the DNS packet and decode its contents
                response = dns.message.from_wire(b64decode(payload))

                # if there is not a question, this is an error and we
                # will write it out to the interesting file.
                #
                # Note: the format here (the addition of the -1) is
                # slightly different than in domainData.data because
                # it has the . for the root
                domain = ".".join(str(response.question[0]).split(".")[8:-1])

                # if we haven't hit this domain yet, then we are
                # getting to data that is not included in
                # domainData.data. We will handle this case by
                # creating a new entry for the domain
                if domain not in domains[ip]:
                    ips, servers = get_ips_and_servers(response)
                    entry = {'timestamp': ts, 'raw_id': raw_id,
                             'resp_time': None, 'ips': ips,
                             'servers': servers}
                else:
                    entry = domains[ip][domain]
                    del domains[ip][domain]
                status = dns.rcode.to_text(response.rcode())
                to_insert.append((ip, domain, entry['resp_time'],
                                  entry['ips'], entry['servers'],
                                  str(response), None, None,
                                  status, entry['timestamp'],
                                  entry["raw_id"], str(response.id)))
                if len(to_insert) > 100000:
                    cursor.executemany(insert_domain, to_insert)
                    db.commit()
                    to_insert = []

                entry['resp'] = response

                # now see if we get different answers for the same question
                question = str(response.question[0])
                # answer = response.answer
                if question in queries[ip]:
                    # payload = b64encode(entry['resp'].to_wire())
                    duplicates.write("{0},{1},{2},{3}"
                                     "\n".format(ip, payload,
                                                 entry['timestamp'],
                                                 entry['raw_id']))
                else:
                    queries[ip][question] = True

            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc(file=PARSE_LOG)
                interesting.write("{0},{1},{2},{3}\n".format(ip, payload,
                                                             ts, raw_id))
                PARSE_LOG.flush()
                interesting.flush()
                continue

    cursor.executemany(insert_domain, to_insert)
    db.commit()
    to_insert = []
    interesting.close()
    PARSE_LOG.close()


def str_to_bool(stringy):
    """Convert the given string to a boolean value"""

    if stringy.lower() == "true":
        return True
    elif stringy.lower() == "false":
        return False
    else:
        ex_mess = ("Invalid boolean option: {}. Need True or "
                   "False".format(stringy))
        raise Exception(ex_mess)


def get_ips_and_servers(response, string_form=True):
    """Pull the resolved IPs out of the answer section of the response"""

    ips = []
    servers = []
    for answer in response.answer:
        if answer.name not in servers:
            servers.append(str(answer.name).strip("."))
        for rdata in answer.items:
            address = rdata.to_text()
            if address not in ips:
                ips.append(address)
    if string_form:
        return ["|".join(ips), '|'.join(servers)]
    else:
        return [ips, servers]


def char_data(domain_file, raw_file, out_file, db_file):
    """Characterize the data and summarize the data in an output file

    We will find:
    1) How many queries had parse issues?
    2) How many queries had duplicates?
    3) What is the distribution of responses (NXDOMAIN vs NOERROR)?
    4) What it the distribution of responses by country?
    5) How many A records did we get?

    You can find how many resolvers responded to the scan and where
    they are located with the charResolvers.py script

    Expected format of domainData.data:
    ip, domain, ts, id, resopnse time, ips, servers, ?, nonce?

    """
    # first, break down responses by whoth

    # part 1- characterize how representative our sample is by looking
    # at who we got responses for each domain
    pass


def explore_duplicates(dup_file, out_file, good_values="",
                       cdn_site_file="../data/cdn-sites.txt"):
    """Explore duplicate responses to see if they are censorship/injection

    We find duplicate responses by finding queries with the same
    question and different answers

    """
    DUP_LOG = open("dup.log", 'w')

    # make a hash table from ips to queries
    queries = {}

    # read in the ips and ases from the control data
    file_p = open(good_values, 'r')
    ctrl_dom2as, ctrl_dom2ip = pickle.load(file_p)
    file_p.close()

    # read in the the cdn sites and mark them if they show up
    cdn_sites = {}
    with open(cdn_site_file, 'r') as file_p:
        for line in file_p.xreadlines():
            cdn_sites[line.strip()] = True

    # create objects for finding the ASNs of IPs
    cc_lookup = GeoIP.open("../data/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)
    cc_reg = re.compile("AS(?P<asn>[0-9]+)")

    # setup the output
    output = open(out_file, 'w')
    fields = ["forwarder", "timestamp", "interarrival", "domain",
              "status1", "status2", "answers1", "answers2",
              "control_answers", "ases1", "ases2", "control_as",
              "asn_is_diff_from_ctrl", "ips_are_diff_from_ctrl",
              "payload1", "payload2", "is_cdn"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    with open(dup_file, 'r') as file_p:
        reader = csv.reader(file_p)
        total = 0
        for row in reader:
            ip, payload, ts, raw_id = row
            entry = {'timestamp': ts, 'raw_id': raw_id, 'payload': payload}

            if ip not in queries:
                queries[ip] = {}

            # decode the DNS packet and decode its contents
            response = b64decode(payload)
            try:
                response = dns.message.from_wire(response)
                entry['resp'] = response

                # now see if we get different answers for the same question
                question = str(response.question[0])
                domain = str(response.question[0].name)
                domain = ".".join(domain.split(".")[8:-1])
                entry['domain'] = domain

                answer = response.answer
                if question in queries[ip]:
                    queries[ip][question].append(entry)
                    for resp in queries[ip][question]:
                        # if the answers are not equal, then this may
                        # be a duplicate
                        #
                        # If it is a duplicate response, give some
                        # basic info, then offer more info for further
                        # analysis
                        if answer != resp['resp'].answer:
                            out_entry = {}
                            out_entry["forwarder"] = ip
                            out_entry["timestamp"] = ts
                            interarrival = float(ts) - float(resp['timestamp'])
                            out_entry["interarrival"] = fabs(interarrival)
                            out_entry['domain'] = domain

                            response2 = resp['resp']
                            rcode1 = response.rcode()
                            rcode2 = response2.rcode()
                            out_entry['status1'] = dns.rcode.to_text(rcode1)
                            out_entry['status2'] = dns.rcode.to_text(rcode2)
                            out_entry['payload1'] = entry['payload']
                            out_entry['payload2'] = resp['payload']

                            out_entry['is_cdn'] = False
                            if domain in cdn_sites:
                                out_entry['is_cdn'] = True

                            # find the IPs and their ASNs and add them into
                            # the vector
                            answers1, asn1 = [], []
                            answers2, asn2 = [], []

                            # go through each IP  and ensure that this
                            # matches the control value.
                            #
                            # There are three options for the resolved
                            # IPs vs the control IPs
                            #
                            # Option 1: the injected response has an
                            # IP address but the control does not. In
                            # this case, we set the initial value to
                            # False (not different) and change it if
                            # we get an IP
                            #
                            # Option 2: the injected and control
                            # responses both have IPs, but they are
                            # different IPs. We deal with this by
                            # setting the initial value to True (is
                            # different) and adjusting if the
                            # responses differ
                            #
                            # Option 3: the control response has an
                            # IP, but the injected response does
                            # not. We deal with this by setting the
                            # initial value to True (is different) and
                            # changing if we get a response

                            out_entry['ips_are_diff_from_ctrl'] = False
                            dom_asn = ctrl_dom2as[domain]
                            out_entry['asn_is_diff_from_ctrl'] = False
                            has_ips = False
                            for part in response.answer:
                                answers1.append(str(part))
                                for item in part.items:
                                    try:
                                        ip = item.address
                                    except AttributeError:
                                        pass
                                    has_ips = True
                                    try:
                                        if ip not in ctrl_dom2ip[domain]:
                                            out_entry['ips_are_diff_from_ctrl'] = True
                                    except KeyError:
                                        pass
                                    as_str = cc_lookup.org_by_addr(ip)
                                    if as_str is None:
                                        continue
                                    match = cc_reg.match(as_str)
                                    if match is None:
                                        continue
                                    asn = match.group('asn')
                                    asn1.append(asn)
                                    if asn != dom_asn:
                                        out_entry['asn_is_diff_from_ctrl'] = True

                            for part in response2.answer:
                                answers2.append(str(part))
                                for item in part.items:
                                    try:
                                        ip = item.address
                                    except AttributeError:
                                        pass
                                        has_ips = True
                                    try:
                                        if ip not in ctrl_dom2ip[domain]:
                                            out_entry['ips_are_diff_from_ctrl'] = True
                                    except KeyError:
                                        pass
                                    as_str = cc_lookup.org_by_addr(ip)
                                    if as_str is None:
                                        continue
                                    match = cc_reg.match(as_str)
                                    if match is None:
                                        continue
                                    asn = match.group('asn')
                                    asn2.append(asn)
                                    if asn != dom_asn:
                                        out_entry['asn_is_diff_from_ctrl'] = True
                            # if the control has IPs/ASN, but neither
                            # response does, then mark the IPs and the
                            # ASN as different
                            if ((not has_ips) and
                               (len(ctrl_dom2ip[domain]) > 0)):
                                out_entry['ips_are_diff_from_ctrl'] = True
                                out_entry['asn_is_diff_from_ctrl'] = True

                            out_entry['answers1'] = "|".join(answers1)
                            out_entry['ases1'] = "|".join(asn1)
                            out_entry['answers2'] = "|".join(answers2)
                            out_entry['ases2'] = "|".join(asn2)
                            out_entry['control_answers'] = "|".join(ctrl_dom2ip[domain].keys())
                            out_entry['control_as'] = ctrl_dom2as[domain]

                            writer.writerow(out_entry)
                            break
                else:
                    queries[ip][question] = [entry]

                total += 1
                if (total % 100000) == 0:
                    print "Finished {}".format(total)
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc(file=DUP_LOG)
                continue

    # with open(interesting, 'w') as output:
    #     for ip in dups:
    #         for question in dups[ip]:
    #             for entry in dups[ip][question]:
    #                 payload = b64encode(entry.resp.to_wire())
    #                 output.write("{0},{1},{2},{3}\n".format(ip, payload,
    #                                                         entry['ts'],
    #                                                         entry['raw_id']))
    output.close()
    DUP_LOG.close()


def find_duplicates(dup_file, raw_file, out_file):
    """Given a duplicate file and a raw file, find the matching packets
    for duplicates and add them all to the out file

    """
    log = open("duplicate-extraction.log", 'w')
    dups = {}
    with open(dup_file, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            ip, payload, ts, raw_id = row

            # decode the DNS packet and decode its contents
            response = b64decode(payload)
            try:
                response = dns.message.from_wire(response)
                question = str(response.question[0])

                if ip not in dups:
                    dups[ip] = {}
                dups[ip][question] = True

            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc(file=log)
                log.flush()
                continue

    output = open(out_file, 'w')
    with open(raw_file, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            ip, payload, ts, raw_id = row

            if ip not in dups:
                continue

            # decode the DNS packet and decode its contents
            response = b64decode(payload)
            try:
                response = dns.message.from_wire(response)
                question = str(response.question[0])

                if question in dups[ip]:
                    output.write("{},{},{},{}\n".format(ip, payload,
                                                        ts, raw_id))

            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc(file=log)
                log.flush()
                continue
    output.close()
    log.close()


def characterize_data(db, basename, out_file="pickled-status-counts.data"):
    """Characterize the dataset for initial analysis

    Questions to answer:
    1) What is the distribution of responses? (NXDOMAIN vs NOERROR,
        etc.) How does this break down by country
    # 2) What is the quality of the responses? How many had parse issues
    #     and how many are duplicates?

    Note: you can separately characterize the resolvers with
    charResolvers.py. That script will answer questions like how many
    resolvers responded and what is their geographic breakdown

    """
    # setup everything that we are going to use
    log = open("char-status.log", 'w')
    db = sqlite3.connect(db, isolation_level="EXCLUSIVE")
    cursor = db.cursor()
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = MEMORY')

    # pull down the resolver and forwarder geoip database
    get_geoip = ("SELECT forwarderIP, forwarderCC, resolverCC "
                 "from forwarderData")
    cursor.execute(get_geoip)
    fwd2cc, fwd2res_cc = {}, {}
    for (fwd_ip, fwd_cc, res_cc) in cursor.fetchall():
        fwd2cc[fwd_ip] = fwd_cc
        fwd2res_cc[fwd_ip] = res_cc

    # Question 1: what is the distribution of responses? How does this
    # break down by country?
    stat2fwd_cnt, stat2res_cnt = {}, {}
    dom2stat2fwd_cnt, dom2stat2res_cnt = {}, {}
    status_patt = re.compile("rcode (?P<rcode>\S+)")
    ccLookup = geoip2.database.Reader('../data/GeoLite2-Country.mmdb')

    # get the data and start iterating

    # Note: we get the data in chunks because Python is bad at
    # buffering, so cursor.fetchall() will get everything and store it
    # in memory, which may be too much on larger datasets
    cursor.execute("SELECT forwarderIP, fullResp, domain from domainData")
    entries = cursor.fetchmany(10000)
    while entries != []:
        for (fwd_ip, fullResp, domain) in entries:
            try:
                # figure out the status
                match = status_patt.search(fullResp)
                if match is None:
                    log.write("Error: no rcode match for {} from {} in:\n"
                              "{}\n".format(fwd_ip, domain, fullResp))
                    continue
                status = match.group('rcode')

                # find the resolver and forwarder country
                if fwd_ip not in fwd2cc:
                    fwd_country = ccLookup.country(fwd_ip).country.iso_code
                    res_country = "unknown"
                else:
                    fwd_country = fwd2cc[fwd_ip]
                    res_country = fwd2res_cc[fwd_ip]

                # do the counting that we came here for
                #
                # start by counting the number of each status from
                # each country
                if status not in stat2fwd_cnt:
                    stat2fwd_cnt[status] = collections.Counter()
                if status not in stat2res_cnt:
                    stat2res_cnt[status] = collections.Counter()
                stat2fwd_cnt[status][fwd_country] += 1
                stat2res_cnt[status][res_country] += 1

                # now count the statuses fro ecah domain
                if domain not in dom2stat2fwd_cnt:
                    dom2stat2fwd_cnt[domain] = {}
                if status not in dom2stat2fwd_cnt[domain]:
                    dom2stat2fwd_cnt[domain][status] = collections.Counter()
                if domain not in dom2stat2res_cnt:
                    dom2stat2res_cnt[domain] = {}
                if status not in dom2stat2res_cnt[domain]:
                    dom2stat2res_cnt[domain][status] = collections.Counter()
                dom2stat2fwd_cnt[domain][status][fwd_country] += 1
                dom2stat2res_cnt[domain][status][res_country] += 1
            except KeyboardInterrupt:
                raise(KeyboardInterrupt)
            except:
                traceback.print_exc(file=log)
                log.flush()

        # get the next set of stuff to iterate over
        entries = cursor.fetchmany(10000)

    # write out the data that we found
    with open(out_file, 'w') as file_p:
        data = {'stat2fwd_cnt': stat2fwd_cnt,
                'stat2res_cnt': stat2res_cnt,
                'dom2stat2fwd_cnt': dom2stat2fwd_cnt,
                'dom2stat2res_cnt': dom2stat2res_cnt}
        pickle.dump(data, file_p)

    # create some graphs of the data
    plot_char_data(basename, stat2fwd_cnt)

    # wrap up
    db.close()
    log.close()


def plot_char_data(basename, in_file=None, stat2fwd_cnt=None,
                   stat2res_cnt=None, dom2stat2fwd_cnt=None,
                   dom2stat2res_cnt=None):
    # if we don't get any data to plot, raise an exception
    if ((in_file is None) and (stat2fwd_cnt is None)):
        raise Exception("You didn't give me any data to plot")

    if in_file:
        with open(in_file, 'r') as file_p:
            data = pickle.load(file_p)
            stat2fwd_cnt = data['stat2fwd_cnt']
            stat2res_cnt = data['stat2res_cnt']
            # dom2stat2fwd_cnt = data['dom2stat2fwd_cnt']
            # dom2stat2res_cnt = data['dom2stat2res_cnt']

    # create some graphs of the data
    # Graph 1: plot the overall distribution of statuses
    statuses = collections.Counter()
    for status in stat2fwd_cnt:
        for country in stat2fwd_cnt[status]:
            statuses[status] += stat2fwd_cnt[status][country]
    filename = basename + "Figure-1-overall-rcode-distribution.pdf"
    total = sum(statuses.values())
    title = ("Overall distribution of all {0} rcodes from a\n"
             "22 Jan 2015 scan".format(total))
    create_pie_chart(statuses, title, filename)

    # Graphs types 2 and 3: plot the distribution of statuses per country
    # (per forwarder and resolver) (this will create a *ton* of graphs)
    countries = {}
    for status in stat2fwd_cnt:
        for country in stat2fwd_cnt[status]:
            if country not in countries:
                countries[country] = True

    # now create a different graph for each country's forwarders
    keys = countries.keys()
    for index in range(len(keys)):
        country = keys[index]
        cnt_statuses = collections.Counter()
        for status in stat2fwd_cnt:
            stat_count = stat2fwd_cnt[status][country]
            cnt_statuses[status] = stat_count
        filename = "".join([basename, "Figure-", str(index),
                            "a-rcode-distribution-for-",
                            str(country), ".pdf"])
        total = sum(cnt_statuses.values())
        title = ("Distribution of all {0} rcodes from forwarders in {1}\n"
                 "during a 22 Jan 2015 scan".format(total, country))
        create_pie_chart(cnt_statuses, title, filename)

    # create the graphs again, but this time for resolvers' geoip
    countries = {}
    for status in stat2res_cnt:
        for country in stat2res_cnt[status]:
            if country not in countries:
                countries[country] = True
    keys = countries.keys()
    for index in range(len(keys)):
        country = keys[index]
        cnt_statuses = collections.Counter()
        for status in stat2res_cnt:
            stat_count = stat2res_cnt[status][country]
            cnt_statuses[status] = stat_count
        filename = "".join([basename, "Figure-", str(index),
                            "b-rcode-distribution-for-",
                            str(country), ".pdf"])
        total = sum(cnt_statuses.values())
        title = ("Distribution of all {0} rcodes from resolvers in {1}\n"
                 "during a 22 Jan 2015 scan".format(total, country))
        create_pie_chart(cnt_statuses, title, filename)


def create_ip_fpr_analysis_file(out_file, good_db, http_db, dns_db,
                                good_values,
                                checkpoint="create-ip-analysis-checkpoint"):
    """Create a CSV file with fingerprints of all the resolved IPs so we
    can decide which fingerprints correspond to censorship and known
    good servers

    The dataset will contain:
    1) are the IPs and ASNs consistent with the control?
    2) are the control IPs and http headers consistent?
    3) are all of the domains part of a CDN? Are any of the domains
        part of a CDN?
    4) How many IPs share the same fingerprint?
    5) What domains resolved to the IP?
    6) How many forwarders is the IP shared across?
    7) How many countries is the IP shared across?
    8) the HTTP fingerprint (headers, server header, body, error)

    Output fields:

    resolved_ip, domains, forwarder_ips, resolver_ips, is_ip_cons,
    is_asn_cons, is_ctrl_server_cons, cdn_domains,
    num_ips_share_same_fpr, num_countries_share_same_fpr, http_headers,
    server_header, http_body, http_error

    """
    fpr_log = open('ip-fpr.log', 'w')
    stage = 0
    data = {}
    if os.path.exists(checkpoint):
        with open(checkpoint, 'r') as file_p:
            stage, data = pickle.load(file_p)

    # read in the ips and ases from the control data
    file_p = open(good_values, 'r')
    ctrl_dom2as, ctrl_dom2ip = pickle.load(file_p)
    file_p.close()

    # create objects to lookup geoip and ASN info
    asn_lookup = GeoIP.open("../data/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)
    asn_reg = re.compile("AS(?P<asn>[0-9]+)")
    cc_lookup = geoip2.database.Reader('../data/GeoLite2-Country.mmdb')

    # create a lookup table from IPs to the domains they resolve to
    print "Creating ip to domain and ip to forwarder lookup tables"
    dns_db = sqlite3.connect(dns_db)
    dns_cursor = dns_db.cursor()
    if stage < 1:
        ip2dom = util.construct_ip_to_site_lookup_table_from_cursor(dns_cursor)
        ip2fwd = util.construct_ip_to_fwd_lookup_table_from_cursor(dns_cursor)
        data = {'ip2dom': ip2dom, 'ip2fwd': ip2fwd}
        stage = 1
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        ip2dom = data['ip2dom']
        ip2fwd = data['ip2fwd']

    # read in the the consistent and inconsistent sites
    with open("analysis-intermediates.data", 'r') as file_p:
        _, analysis_data = pickle.load(file_p)
    # goodIPs = analysis_data['goodIPs']
    # consIPSites = analysis_data['consIPSites']
    cdn_sites = analysis_data['inconIPSites']

    # read in the control HTTP headers
    good_db = sqlite3.connect(good_db)
    good_db.text_factory = str
    cursor = good_db.cursor()
    print "Reading in good HTTP data"
    good_data = {}
    if stage < 2:
        cursor.execute(("SELECT site, ips, ipCons, headers, body, status, "
                        "error, "
                        "serverHeader, consServer, consHeaderOrder, "
                        "consHeaderVal, consError from goodData"))
        for entry in cursor.fetchall():
            site, ips, ip_cons, headers, body, status, error = entry[:7]
            serv_head, cons_serv, cons_head_ord, cons_head_val = entry[7:11]
            cons_err = entry[-1]
            good_data[site] = {'ips': ips, 'ip_cons': ip_cons,
                               'header': headers, 'body': body,
                               'status': status, 'error': error,
                               'serv_head': serv_head, 'cons_serv':
                               cons_serv, 'cons_head_ord':
                               cons_head_ord, 'cons_head_val':
                               cons_head_val, 'cons_err': cons_err}
        stage = 2
        data['good_data'] = good_data
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        good_data = data['good_data']

    # read in the other HTTP headers
    print "Reading in HTTP fingerprinting data"
    http_db = sqlite3.connect(http_db)
    http_db.text_factory = str
    http_cursor = http_db.cursor()
    http_query = ("SELECT ip, error, status, headers, server, body FROM "
                  "webServers")
    http_cursor.execute(http_query)
    http_data = {}
    servers = {}
    if stage < 3:
        for (ip, error, status, headers, server, body) in http_cursor.fetchall():
            http_data[ip] = {'error': str(error), 'status': str(status),
                             'headers': str(headers),
                             'server': str(server), 'body': str(body)}
            if server not in servers:
                servers[server] = [ip]
            else:
                servers[server].append(ip)
        stage = 3
        data['http_data'] = http_data
        data['servers'] = servers
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        http_data = data['http_data']
        servers = data['servers']

    output = open(out_file, 'w')
    fields = ['resolved_ip', 'domains', 'forwarder_ips',
              'num_forwarder_ips', 'is_cdn', 'is_asn_cons',
              'is_ctrl_server_cons', 'is_serv_cons_with_ctrl',
              'num_ips_share_same_fpr', 'ips_share_same_fpr',
              'fwd_countries_share_same_fpr',
              'num_fwd_countries_share_same_fpr', 'http_headers',
              'server_header', 'http_body', 'http_error']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    print "Starting analysis"
    # iterate over the IPs and output an entry for each distinct
    # server header (this is the extent of HTTP fingerprinting for now
    total = 0
    for server in servers.keys():
        ips = servers[server]
        countries, forwarders = {}, {}
        to_write = []
        total += 1
        if (total % 50) == 0:
            print "Completed {}/{} server headers".format(total,
                                                          len(servers.keys()))
        for ip in servers[server]:
            try:
                domains = ip2dom[ip]
                http_stuff = http_data[ip]

                # figure out which country we got this ip from
                fwds = ip2fwd[ip]
                for fwd in fwds:
                    country = cc_lookup.country(fwd).country.iso_code
                    if country is None:
                        country = "unknown"
                    countries[country] = True
                    forwarders[fwd] = True

                # lookup the asn
                as_str = asn_lookup.org_by_addr(ip)
                if as_str is None:
                    asn = ""
                else:
                    match = asn_reg.match(as_str)
                    if match is None:
                        asn = ""
                    else:
                        asn = match.group('asn')

                # if any of the domains are inconsistent, then mark the AS
                # as inconsisent
                is_cdn, asn_cons = False, True
                ctrl_serv_cons, diff_serv = True, False
                for domain in domains:
                    # check if we are on a CDN
                    if domain in cdn_sites:
                        is_cdn = True

                    if domain in good_data:
                        good_entry = good_data[domain]
                        # check if the control entries were consistent
                        if good_entry['cons_serv'] is False:
                            ctrl_serv_cons = False
                        # check if the server header is consistent with the
                        # control for each domain
                        if (good_entry['serv_head'] != server):
                            diff_serv = True
                    else:
                        if (server != "") or (server is not None):
                            diff_serv = True
                            ctrl_serv_cons = False

                    # check that the asn is consistent
                    if domain not in ctrl_dom2as:
                        asn_cons = False
                    elif ctrl_dom2as[domain] != asn:
                        asn_cons = False

                # if the IPs or the server header is consistent with
                # the control, then don't print anything
                if asn_cons or (not diff_serv):
                    continue

                # if we already know that this is censorship, then
                # don't print it either
                if "norton.com" in body:
                    continue

                headers = http_stuff['headers']
                headers = headers.replace("\r\n", "|")
                headers = headers.replace("\n", "|")
                body = http_stuff['body']
                body = body.replace("\n", "|")
                domains = "|".join(domains)
                entry = {'resolved_ip': ip, 'domains': domains,
                         'is_cdn': is_cdn, 'is_asn_cons': asn_cons,
                         'is_ctrl_server_cons': ctrl_serv_cons,
                         'is_serv_cons_with_ctrl': diff_serv,
                         'num_ips_share_same_fpr': len(ips),
                         'ips_share_same_fpr': "|".join(ips),
                         'http_headers': headers,
                         'server_header': http_stuff['server'],
                         'http_body': body,
                         'http_error': http_stuff['error']}
                to_write.append(entry)
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc(file=fpr_log)
        cnts_same_fpr = "|".join(sorted(countries.keys()))
        num_cnts_same_fpr = len(cnts_same_fpr)
        num_forwarders = len(forwarders.keys())
        forwarders = "|".join(forwarders.keys())
        for entry in to_write:
            entry['fwd_countries_share_same_fpr'] = cnts_same_fpr
            entry['num_fwd_countries_share_same_fpr'] = num_cnts_same_fpr
            entry['forwarder_ips'] = forwarders
            entry['num_forwarder_ips'] = num_forwarders
            writer.writerow(entry)
    output.close()
    fpr_log.close()


def find_censorship_raw_nxdomain(raw_file, dns_db, web_db, results_db,
                                 categories, mon_control, good_values,
                                 results_summary, ptr_file,
                                 checkpoint="summary-analysis-checkpoint"):
    """Find censorship in the overall results database

    Note: this does a ton of analysis on top of the raw DNS results


    Note: this will create a single file with the consolidated
    censorship results

    We determine if censorship is taking place by checking:

    Step 1) check if this is a normal response by seeing if the IPs or
       ASNs are consistent with the control. If they differ and the
       control values were consistent, then this is manipulation.

    Step 2) check if the IP is on a CDN by checking for server
       consistency and checking the header against known CDNs

    Step 3) check if this is censorship or monetization by comparing
       the resolved IPs to our control NXDOMAIN measurement and by
       checking if that forwarder has a history of NXDOMAIN

    Format of results_summary:
    forwarder, domain, isIPBlock, isDNSBlock, fwd_cc, categories

    """
    # read in any previous checkpointing data
    stage, data = 0, {}
    if os.path.exists(checkpoint):
        with open(checkpoint, 'r') as file_p:
            stage, data = pickle.load(file_p)

    print "Reading in the CDN domains and IP consistency stuff"
    with open("analysis-intermediates.data", 'r') as file_p:
        _, analysis_data = pickle.load(file_p)
    # goodIPs = analysis_data['goodIPs']
    # consIPSites = analysis_data['consIPSites']
    cdn_sites = analysis_data['inconIPSites']
    # httpCons = data['httpCons']
    # goodHTTP = data['goodHTTP']

    # read in the ips and ases from the control data
    file_p = open(good_values, 'r')
    ctrl_dom2as, ctrl_dom2ip = pickle.load(file_p)
    file_p.close()

    # create the maxmind lookup object
    cc_lookup = geoip2.database.Reader('../data/GeoLite2-Country.mmdb')

    # create objects to lookup ASN info
    asn_lookup = GeoIP.open("../data/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)
    asn_reg = re.compile("AS(?P<asn>[0-9]+)")

    dom2cat = {}
    # read in the information about what categories each domain
    # belongs to
    with open(categories, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            domain, cats = row
            cats = cats.split("|")
            for cat in cats:
                if domain not in dom2cat:
                    dom2cat[domain] = [cat]
                else:
                    if cat not in dom2cat[domain]:
                        dom2cat[domain].append(cat)

    # read in the control NXDOMAIN measurement info
    monet_ips, fwd2mon, fwd_has_mon = {}, {}, {}
    with open(mon_control, 'r') as file_p:
        # read the first line to get rid of the header
        file_p.readline()
        for line in file_p.xreadlines():
            fwd, has_monet, mon_ips, mon_names, stat = line.strip().split(",")
            fwd_has_mon[fwd] = str_to_bool(has_monet)
            if not has_monet:
                continue

            mon_ips = mon_ips.split("|")
            for ip in mon_ips:
                monet_ips[ip] = True
                if fwd in fwd2mon:
                    fwd2mon[fwd].append(ip)
                else:
                    fwd2mon[fwd] = [ip]

    web_db = sqlite3.connect(web_db)
    web_db.text_factory = str
    res_db = sqlite3.connect(results_db)
    res_db.text_factory = str
    dns_db = sqlite3.connect(dns_db)
    dns_db.text_factory = str
    web_cursor = web_db.cursor()
    # res_cursor = res_db.cursor()
    dns_cursor = dns_db.cursor()

    print "Creating ip to domain and ip to forwarder lookup tables"
    if stage < 1:
        ip2dom = util.construct_ip_to_site_lookup_table_from_cursor(dns_cursor)
        ip2fwd = util.construct_ip_to_fwd_lookup_table_from_cursor(dns_cursor)
        data = {'ip2dom': ip2dom, 'ip2fwd': ip2fwd}
        stage = 1
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        ip2dom = data['ip2dom']
        ip2fwd = data['ip2fwd']

    print "Fetching good HTTP header data"
    if stage < 2:
        data1 = interference.get_http_responses(cdn_sites.keys(),
                                                nxdomain=True)
        data2 = interference.get_http_responses(cdn_sites.keys(),
                                                nxdomain=True)
        good_data = {}
        for site in data1.keys():
            good_data[site] = interferenceAnalysis.check_http_header_consistency(data1[site]['headers'], data2[site]['headers'])
            good_data[site].update(data1[site])

        stage = 2
        data['good_data'] = good_data
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        good_data = data['good_data']

    # read in the other HTTP headers
    print "Reading in HTTP fingerprinting data"
    http_data, servers = {}, {}
    if stage < 3:
        web_query = ("SELECT ip, error, status, headers, server, body FROM "
                     "webServers")
        web_cursor.execute(web_query)
        for (ip, err, status, headers, server, body) in web_cursor.fetchall():
            http_data[ip] = {'error': str(err), 'status': str(status),
                             'headers': str(headers),
                             'server': str(server), 'body': str(body)}
            if server not in servers:
                servers[server] = [ip]
            else:
                servers[server].append(ip)
        stage = 3
        data['http_data'] = http_data
        data['servers'] = servers
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        http_data = data['http_data']
        servers = data['servers']

    # read in the ptr data
    ptrs = {}
    with open(ptr_file, 'r') as file_p:
        for line in file_p.readlines():
            entry = json.loads(line)
            ptrs[entry['query']] = entry

    # apache/2.2.27 giving a lot of 404s, about.com's server (with
    # redirects) is Jetty without is nginx_about/1.4.2_5
    # monetization_serv_heads = {"Apache/2.2.27": True,
    #                            "Jetty(9.0.z-SNAPSHOT)": True}
    # censor_serv_heads = {}

    # pull the rest of the db and determine if censorship is
    # happening. We determine if censorship is happening by going
    # through 3 steps
    #
    # Step 1: check if this is an expected response (i.e. that IP/ ASN
    # is consistent)
    #
    # Step 2: check if this is a valid CDN IP by checking for control
    # consistency and checking the header against the control
    #
    # Step 3: now that we know we have manipulation, check if this is
    # monetization by a) checking the resolved IPs against a list of
    # monetization IPs and b) checking if that resolver has a history
    # of monetization
    output = open(results_summary, 'w')
    fields = ['forwarder', 'domain', 'isIPBlock', 'isDNSBlock',
              'fwd_cc', 'categories', 'status', 'reason', 'ips',
              'asn', 'server', 'ptrs']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    sum_log = open("summary.log", 'w')
    unclear_http = open('http-diff-serv.log', 'w')
    odd_monet = open("monetization.log", 'w')
    parse_log = open("parse-log.log", 'w')
    bad_status_log = open("bad-status.log", 'w')
    odd_resp_log = open("odd-dns-response.log", 'w')

    input_f = open(raw_file, 'r')
    reader = csv.reader(input_f)
    print "Starting to process results"
    for row in reader:
        fwd, payload, timestamp, rawID = row
        entry = {'forwarder': fwd}
        try:
            # decode the DNS packet and decode its contents
            resp = dns.message.from_wire(b64decode(payload))
        except KeyboardInterrupt:
            exit(0)
        except:
            parse_log.write("{},{},{},{}\n".format(fwd, payload, timestamp,
                                                   rawID))

        # get a bunch of information from the DNS response
        try:
            # Note: the format here (the addition of the -1) is
            # slightly different than in domainData.data because
            # it has the . for the root
            has_subdom = True
            domain = ".".join(str(resp.question[0]).split(".")[8:-1])
            if domain == "":
                has_subdom = False
                domain = str(resp.question[0])
            entry['domain'] = domain

            # get the status code of the response (rcode)
            status = dns.rcode.to_text(resp.rcode())
            entry['status'] = status

            # pull out the ips, servers
            ips, names = get_ips_and_servers(resp, string_form=False)

            # flags = dns.flags.to_text(resp.flags)
            # can_resolve = ("RA" in flags)

            # if the status is anything other than NOERROR, then we
            # won't consider this for censorship. However, we also
            # don't want to fill up the log with NXDOMAIN records (80%
            # of responses), so only write out if we are not NXDOMAIN.
            #
            # also limit what we mark as censorship by trashing
            # anything without recursion desired flag set
            #
            # if we don't have a domain, then also drop it because we
            # don't have any way of seeing what is censored
            # if ((not can_resolve) or (not has_subdom) or
            if ((not has_subdom) or
               ((status != "NOERROR") and (status != "NXDOMAIN"))):
                bad_status_log.write("{},{},{},{}\n".format(fwd, payload,
                                                            timestamp,
                                                            rawID))
                continue

        except KeyboardInterrupt:
            exit(0)
        except:
            odd_resp_log.write("{},{},{},{}\n".format(fwd, payload, timestamp,
                                                      rawID))

        # get the rest of the info
        try:
            # lookup the country of the forwarder
            fwd_cc = cc_lookup.country(fwd).country.iso_code
            entry['fwd_cc'] = fwd_cc

            # lookup the categories for this domain
            cats = "|".join(dom2cat[domain])
            entry['categories'] = cats
            entry['isIPBlock'] = ""
            entry['ips'] = "|".join(ips)

            # get information about each ip
            serv_heads, asns, ptr_recs = [], [], []
            if len(ips) == 0:
                asns = [""]
            for ip in ips:
                # get the server header for this IP if it is available
                if ip in http_data:
                    http_entry = http_data[ip]
                    serv_head = http_entry['server']
                    serv_heads.append(serv_head)

                # lookup the asn for each IP
                asn = ""
                if ip != "":
                    as_str = asn_lookup.org_by_addr(ip)
                    if as_str is not None:
                        match = asn_reg.match(as_str)
                        if match is not None:
                            asn = match.group('asn')
                asns.append(asn)

                # lookup the ptr record for each ip
                if ip in ptrs:
                    ptr_ent = ptrs[ip]
                    if ptr_ent['status'] == "NXNAME":
                        ptr_recs.append("NXDOMAIN")
                    elif ptr_ent['status'] != "NOERROR":
                        ptr_recs.append(ptr_ent['status'])
                    else:
                        ptr_recs.append(ptr_ent['values']['answer'][0][3])
            entry['ptrs'] = "|".join(ptr_recs)
            entry['asn'] = "|".join(asns)
            entry['server'] = "|".join(serv_heads)

            # mark this is non-censored if we get an nxdomain response
            if status == "NXDOMAIN":
                entry['isDNSBlock'] = False
                entry['reason'] = 'status_nxdomain'
                writer.writerow(entry)
                continue

            # step 1: if the IPs/ASNs are consistent, then this is not
            # censorship
            #
            # lookup the control ASN for this domain
            dom_asn = {"": True}
            if domain in ctrl_dom2as:
                dom_asn = ctrl_dom2as[domain]
            # check for consistent ASNs and if not, then keep going to
            # the other options
            for asn in asns:
                if asn in dom_asn:
                    entry['isDNSBlock'] = False
                    entry['reason'] = 'asn_same'
                    writer.writerow(entry)
                    # we use a break here so that we break out, then
                    # the continue pushes us to the next entry
                    break
            # continue if the asn is equal to dom_asn because we don't
            # want duplicate entries
            if asn in dom_asn:
                continue

            # step 2: if the site is on a CDN (control had
            #    inconsistent resolved IPs) and the server header
            #    matches, then this is not censorship
            ctrl_serv_head = ""
            if domain in good_data:
                ctrl_serv_head = good_data[domain]['serverHeader']
            elif domain in ctrl_dom2ip:
                for ip in ctrl_dom2ip[domain]:
                    if ip in http_data:
                        ctrl_serv_head = http_data[ip]['server']
            diff_serv = False
            if (len(serv_heads) == 0):
                if (ctrl_serv_head == "") or (ctrl_serv_head is None):
                    diff_serv = True
            # for each server header, see if it compares
            # if domain in cdn_sites:
            # validate that the server header is consistent
            for serv_head in serv_heads:
                if serv_head != ctrl_serv_head:
                    # handle the case where the control header is
                    # consistently None or "" but we get a
                    # monetization header
                    if serv_head == "" and ctrl_serv_head is None:
                        continue
                    diff_serv = True
            # if the control header was consistent and we
            # didn't match the header, then this is censorship
            if diff_serv:
                # check to see if the control header was consistent
                if ((domain in good_data) and
                   (good_data[domain]['consServer'])):
                    entry['reason'] = 'diff_server'
                    entry['isDNSBlock'] = True
                    writer.writerow(entry)
                    continue
                unclear_http.write("{},{},{},{}\n".format(fwd, payload,
                                                          timestamp, rawID))
            else:
                # if the control header was consistent and we
                # matched the header, then this is *NOT* censorhsip
                entry['reason'] = 'same_server'
                entry['isDNSBlock'] = False
                writer.writerow(entry)
                continue

            # step 3: check if this is censorship or monetization by
            # comparing the resolved IPs against a list of known
            # monetization servers and seeing if the resolver has a
            # history of monetization
            for ip in ips:
                is_monet = (ip in monet_ips)
                fwd_has_monet = True
                if ip in ip2fwd:
                    for forwarder in ip2fwd[ip]:
                        if forwarder not in fwd_has_mon:
                            fwd_has_monet = False
                # if the IP is not in the ip2fwd table, try the
                # orginal IP, then mark it
                else:
                    if fwd not in fwd_has_mon:
                        fwd_has_monet = False
                # if both of these are not True or false, we have an
                # odd situation because the resolved IP address
                # belongs to a monetization server, but this resolver
                # didn't give us monetization stuff for the control
                if is_monet and not fwd_has_monet:
                    odd_monet.write("Odd situation: is_monet: {} "
                                    "fwd_has_monet: {} " "for {} and {} "
                                    "with resolved IP {}"
                                    "\n".format(is_monet, fwd_has_monet,
                                                fwd, domain, ip))
                # if we have a consensus that this is monetization, then
                # mark this as non-censored
                if is_monet:
                    entry['reason'] = 'monet'
                    entry['isDNSBlock'] = False
                    writer.writerow(entry)
                    break
            # also continue here so that we don't get duplicates in
            # the not censored file and we don't treat this as
            # censorship
            if is_monet:
                continue
            # if this is not monetization and we have gotten this far,
            # then this is censorship
            entry['reason'] = 'nothing_left'
            entry['isDNSBlock'] = True
            writer.writerow(entry)
            continue

        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc(file=sum_log)

    output.close()
    parse_log.close()
    odd_resp_log.close()
    bad_status_log.close()
    odd_monet.close()
    sum_log.close()


def find_censorship_raw(raw_file, dns_db, web_db, categories,
                        mon_control, good_values, results_summary,
                        ptr_file, checkpoint="summary-analysis-checkpoint"):
    """Find censorship in the overall results database

    Note: this does a ton of analysis on top of the raw DNS results

    Note: this will create a single file with the consolidated
    censorship results

    We determine if censorship is taking place by checking:

    Step 1) check if this is a normal response by seeing if the IPs or
       ASNs are consistent with the control. If they differ and the
       control values were consistent, then this is manipulation.

    Step 2) check if the IP is on a CDN by checking for server
       consistency and checking the header against known CDNs

    Format of results_summary:
    forwarder, domain, isIPBlock, isDNSBlock, fwd_cc, categories,
    status, recurAvail, reason, ips, asn, server, ptrs

    """
    # read in any previous checkpointing data
    stage, data = 0, {}
    if os.path.exists(checkpoint):
        with open(checkpoint, 'r') as file_p:
            stage, data = pickle.load(file_p)

    print "Reading in the CDN domains and IP consistency stuff"
    with open(good_values, 'r') as file_p:
        good_data = pickle.load(file_p)
        ctrl_dom2as = good_data['dom2asn']
        ctrl_dom2ip = good_data['dom2ip']
        cdn_sites = good_data['cdnSites']

    # create the maxmind lookup object
    cc_lookup = geoip2.database.Reader('../data/GeoLite2-Country.mmdb')

    # create objects to lookup ASN info
    asn_lookup = GeoIP.open("../data/GeoIPASNum.dat", GeoIP.GEOIP_STANDARD)
    asn_reg = re.compile("AS(?P<asn>[0-9]+)")

    dom2cat = {}
    # read in the information about what categories each domain
    # belongs to
    with open(categories, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            domain, cats = row
            cats = cats.split("|")
            for cat in cats:
                if domain not in dom2cat:
                    dom2cat[domain] = [cat]
                else:
                    if cat not in dom2cat[domain]:
                        dom2cat[domain].append(cat)

    # read in the control NXDOMAIN measurement info
    monet_ips, fwd2mon, fwd_has_mon = {}, {}, {}
    with open(mon_control, 'r') as file_p:
        # read the first line to get rid of the header
        file_p.readline()
        for line in file_p.xreadlines():
            fwd, has_monet, mon_ips, mon_names, stat = line.strip().split(",")
            fwd_has_mon[fwd] = str_to_bool(has_monet)
            if not has_monet:
                continue

            mon_ips = mon_ips.split("|")
            for ip in mon_ips:
                monet_ips[ip] = True
                if fwd in fwd2mon:
                    fwd2mon[fwd].append(ip)
                else:
                    fwd2mon[fwd] = [ip]

    web_db = sqlite3.connect(web_db)
    web_db.text_factory = str
    dns_db = sqlite3.connect(dns_db)
    dns_db.text_factory = str
    web_cursor = web_db.cursor()
    dns_cursor = dns_db.cursor()

    print "Creating ip to domain and ip to forwarder lookup tables"
    if stage < 1:
        ip2dom = util.construct_ip_to_site_lookup_table_from_cursor(dns_cursor)
        ip2fwd = util.construct_ip_to_fwd_lookup_table_from_cursor(dns_cursor)
        data = {'ip2dom': ip2dom, 'ip2fwd': ip2fwd}
        stage = 1
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        ip2dom = data['ip2dom']
        ip2fwd = data['ip2fwd']

    print "Fetching good HTTP header data"
    if stage < 2:
        data1 = interference.get_http_responses(cdn_sites.keys())
        data2 = interference.get_http_responses(cdn_sites.keys())
        good_data = {}
        for site in data1.keys():
            good_data[site] = interferenceAnalysis.check_http_header_consistency(data1[site]['headers'], data2[site]['headers'])
            good_data[site].update(data1[site])

        stage = 2
        data['good_data'] = good_data
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        good_data = data['good_data']

    # read in the other HTTP headers
    print "Reading in HTTP fingerprinting data"
    http_data, servers = {}, {}
    if stage < 3:
        web_query = ("SELECT ip, error, status, headers, server, body FROM "
                     "webServers")
        web_cursor.execute(web_query)
        for (ip, err, status, headers, server, body) in web_cursor.fetchall():
            http_data[ip] = {'error': str(err), 'status': str(status),
                             'headers': str(headers),
                             'server': str(server), 'body': str(body)}
            if server not in servers:
                servers[server] = [ip]
            else:
                servers[server].append(ip)
        stage = 3
        data['http_data'] = http_data
        data['servers'] = servers
        with open(checkpoint, 'w') as file_p:
            pickle.dump([stage, data], file_p)
    else:
        http_data = data['http_data']
        servers = data['servers']

    # read in the ptr data
    ptrs = {}
    with open(ptr_file, 'r') as file_p:
        for line in file_p.readlines():
            entry = json.loads(line)
            ptrs[entry['query']] = entry

    # pull the rest of the db and determine if censorship is
    # happening. We determine if censorship is happening by going
    # through 2 steps
    #
    # Step 1: check if this is an expected response (i.e. that IP/ ASN
    # is consistent)
    #
    # Step 2: check if this is a valid CDN IP by checking for control
    # consistency and checking the header against the control
    output = open(results_summary, 'w')
    fields = ['forwarder', 'domain', 'isIPBlock', 'isDNSBlock',
              'fwd_cc', 'categories', 'status', 'recurAvail', 'reason', 'ips',
              'asn', 'server', 'ptrs']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    sum_log = open("summary.log", 'w')
    unclear_http = open('http-diff-serv.log', 'w')
    parse_log = open("parse-log.log", 'w')
    bad_status_log = open("bad-status.log", 'w')
    bad_resolve_log = open("recur-not-avail.log", 'w')
    odd_resp_log = open("odd-dns-response.log", 'w')

    input_f = open(raw_file, 'r')
    reader = csv.reader(input_f)
    print "Starting to process results"
    for row in reader:
        fwd, payload, timestamp, rawID = row
        entry = {'forwarder': fwd}
        try:
            # decode the DNS packet and decode its contents
            resp = dns.message.from_wire(b64decode(payload))
        except KeyboardInterrupt:
            exit(0)
        except:
            parse_log.write("{},{},{},{}\n".format(fwd, payload, timestamp,
                                                   rawID))

        # get a bunch of information from the DNS response
        try:
            # if this is a tagged query or we don't have a domain,
            # then trash it
            domain = str(resp.question[0].name)
            domain = domain.strip(".")
            entry['domain'] = domain
            if (len(domain.split(".")) > 9) or (domain == ""):
                continue

            # get the status code of the response (rcode)
            status = dns.rcode.to_text(resp.rcode())
            entry['status'] = status

            # pull out the ips, servers
            ips, names = get_ips_and_servers(resp, string_form=False)

            flags = dns.flags.to_text(resp.flags)
            can_resolve = ("RA" in flags)
            entry["recurAvail"] = can_resolve
            if not can_resolve:
                bad_resolve_log.write("{},{},{},{}\n".format(fwd, payload,
                                                             timestamp,
                                                             rawID))
                continue

            # if the status is NXDOMAIN, then we consider it
            # censorship, and we could consider anything that is not
            # an NOERROR response as censorship, but we don't do this
            # to account for unexpected problems/errors
            #
            # also limit what we mark as censorship by trashing
            # anything without recursion desired flag set
            #
            # if we don't have a domain, then also drop it because we
            # don't have any way of seeing what is censored
            # if ((not can_resolve) or (not has_subdom) or
            if status == "NXDOMAIN":
                bad_status_log.write("{},{},{},{}\n".format(fwd, payload,
                                                            timestamp,
                                                            rawID))
                continue

        except KeyboardInterrupt:
            exit(0)
        except:
            odd_resp_log.write("{},{},{},{}\n".format(fwd, payload, timestamp,
                                                      rawID))
            continue

        # get the rest of the info
        try:
            # lookup the country of the forwarder
            fwd_cc = cc_lookup.country(fwd).country.iso_code
            entry['fwd_cc'] = fwd_cc

            # lookup the categories for this domain
            cats = "|".join(dom2cat[domain])
            entry['categories'] = cats
            entry['isIPBlock'] = ""
            entry['ips'] = "|".join(ips)

            # get information about each ip
            serv_heads, asns, ptr_recs = [], [], []
            ips_are_consistent = True
            if len(ips) == 0:
                asns = [""]
                ips_are_consistent = False
            for ip in ips:
                # get the server header for this IP if it is available
                if ip in http_data:
                    http_entry = http_data[ip]
                    serv_head = http_entry['server']
                    serv_heads.append(serv_head)

                # lookup the asn for each IP
                asn = ""
                if ip != "":
                    as_str = asn_lookup.org_by_addr(ip)
                    if as_str is not None:
                        match = asn_reg.match(as_str)
                        if match is not None:
                            asn = match.group('asn')
                asns.append(asn)

                # see if the ip is in the list for this domain
                if ip not in ctrl_dom2ip[domain]:
                    ips_are_consistent = False

                # lookup the ptr record for each ip
                if ip in ptrs:
                    ptr_ent = ptrs[ip]
                    if ptr_ent['status'] == "NXNAME":
                        ptr_recs.append("NXDOMAIN")
                    elif ptr_ent['status'] != "NOERROR":
                        ptr_recs.append(ptr_ent['status'])
                    else:
                        ptr_recs.append(ptr_ent['values']['answer'][0][3])
            entry['ptrs'] = "|".join(ptr_recs)
            entry['asn'] = "|".join(asns)
            entry['server'] = "|".join(serv_heads)

            if ips_are_consistent:
                entry['isDNSBlock'] = False
                entry['reason'] = 'ips_same'
                writer.writerow(entry)

            # step 1: if the IPs/ASNs are consistent, then this is not
            # censorship
            #
            # lookup the control ASN for this domain
            dom_asn = {"": True}
            if domain in ctrl_dom2as:
                dom_asn = ctrl_dom2as[domain]
            # check for consistent ASNs and if not, then keep going to
            # the other options
            for asn in asns:
                if asn in dom_asn:
                    entry['isDNSBlock'] = False
                    entry['reason'] = 'asn_same'
                    writer.writerow(entry)
                    # we use a break here so that we break out, then
                    # the continue pushes us to the next entry
                    break
            # continue if the asn is equal to dom_asn because we don't
            # want duplicate entries
            if asn in dom_asn:
                continue

            # step 2: if the site is on a CDN (control had
            #    inconsistent resolved IPs) and the server header
            #    matches, then this is not censorship
            ctrl_serv_head = ""
            if domain in good_data:
                ctrl_serv_head = good_data[domain]['serverHeader']
            elif domain in ctrl_dom2ip:
                for ip in ctrl_dom2ip[domain]:
                    if ip in http_data:
                        ctrl_serv_head = http_data[ip]['server']
            diff_serv = False
            if (len(serv_heads) == 0):
                if (ctrl_serv_head == "") or (ctrl_serv_head is None):
                    diff_serv = True
            # for each server header, see if it compares
            # if domain in cdn_sites:
            # validate that the server header is consistent
            for serv_head in serv_heads:
                if serv_head != ctrl_serv_head:
                    # handle the case where the control header is
                    # consistently None or "" but we get a
                    # monetization header
                    if serv_head == "" or serv_head is None:
                    # if serv_head == "" and ctrl_serv_head is None:
                        continue
                    diff_serv = True
            # if the control header was consistent and we
            # didn't match the header, then this is censorship
            if diff_serv:
                # check to see if the control header was consistent
                if ((domain in good_data) and
                   (good_data[domain]['consServer'])):
                    entry['reason'] = 'diff_server'
                    entry['isDNSBlock'] = True
                    writer.writerow(entry)
                    continue
                unclear_http.write("{},{},{},{}\n".format(fwd, payload,
                                                          timestamp, rawID))
            else:
                # if the control header was consistent and we
                # matched the header, then this is *NOT* censorhsip
                entry['reason'] = 'same_server'
                entry['isDNSBlock'] = False
                writer.writerow(entry)
                continue

            # if we have gotten this far, then this is censorship
            entry['reason'] = 'nothing_left'
            entry['isDNSBlock'] = True
            writer.writerow(entry)
            continue

        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc(file=sum_log)

    output.close()
    parse_log.close()
    odd_resp_log.close()
    bad_status_log.close()
    bad_resolve_log.close()
    sum_log.close()


def add_info_to_censorship_file(in_file, out_file, categories, fwd_file):
    """Given a file with the forwarder and censored domain, add the
    resolver, the geoip lookups for the forwarder and the resolver,
    and the categories of the domain

    """
    output = open(out_file, 'w')
    fields = ["forwarder", "resolver", "domain", "fwd_cc", "res_cc",
              "categories"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    fwd_info = {}
    # read in the forwarder data
    with open(fwd_file, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            fwd_ip, res_ip, resp_time, ts, fwd_cc, res_cc = row
            fwd_info[fwd_ip] = {'res_ip': res_ip, 'fwd_cc': fwd_cc,
                                'res_cc': res_cc}

    dom2cat = {}
    # read in the information about what categories each domain
    # belongs to
    with open(categories, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            domain, cats = row
            cats = cats.split("|")
            for cat in cats:
                if domain not in dom2cat:
                    dom2cat[domain] = [cat]
                else:
                    if cat not in dom2cat[domain]:
                        dom2cat[domain].append(cat)

    # now read in the input file and add this new information
    ccLookup = geoip2.database.Reader('../data/GeoLite2-Country.mmdb')
    with open(in_file, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            try:
                fwd, domain = row
            except ValueError:
                print row
                traceback.print_exc()
            entry = {'forwarder': fwd, "domain": domain}
            if fwd in fwd_info:
                fwd_entry = fwd_info[fwd]
                entry["resolver"] = fwd_entry["res_ip"]
                entry["fwd_cc"] = fwd_entry["fwd_cc"]
                entry["res_cc"] = fwd_entry["res_cc"]
            else:
                entry["fwd_cc"] = ccLookup.country(fwd).country.iso_code
                entry['res_cc'] = "unknown"
            entry["categories"] = "|".join(dom2cat[domain])
            writer.writerow(entry)


def find_monetization(raw_file, out_file):
    """Given the raw responses, pull out all of the queries to
    <nonce>.www.cs.princeton.edu that resolve to something other than
    an NXDOMAIN

    Output file format:
    forwarder, has_monetization, monetization_servers

    """
    nxdom_log = open("nxdomain-analysis.log", 'w')
    output = open(out_file, 'w')
    fields = ["forwarder", 'has_monetization', 'monetization_ips',
              'monetization_names', 'status']
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()

    with open(raw_file, 'r') as file_p:
        reader = csv.reader(file_p)
        for row in reader:
            fwd, payload, ts, raw_id = row
            entry = {'forwarder': fwd}

            try:
                # decode the DNS packet and decode its contents
                response = dns.message.from_wire(b64decode(payload))

                # Note: the format here (the addition of the -1) is
                # slightly different than in domainData.data because
                # it has the . for the root
                domain = ".".join(str(response.question[0]).split(".")[8:-1])
                if domain != "nxdomain-test.server.netalyzr.icsi.berkeley.edu":
                    continue

                entry['has_monetization'] = False
                status = dns.rcode.to_text(response.rcode())
                if status == "NOERROR":
                    entry['has_monetization'] = True

                # now find the resolved IPs
                ips, names = [], []
                for answer in response.answer:
                    name = str(answer.name)
                    if name not in names:
                        names.append(name)
                    for item in answer.items:
                        if item.address not in ips:
                            ips.append(item.address)
                entry['monetization_ips'] = "|".join(ips)
                entry['monetization_names'] = "|".join(names)
                entry['status'] = status
                writer.writerow(entry)
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc(file=nxdom_log)

    output.close()
    nxdom_log.close()


def create_graphs(in_file, basename,
                  dom_cnt_mat_file="../data/dom-cnt-matrix.data"):

    """Given an input file according to the format output by
    find_censorship, this will generate a bunch of graphs
    to analyze censorship and at a higher level, how topics are
    treated within each country

    Expected input format:
    forwarder, domain, isIPBlock, isDNSBlock, fwd_cc, categories

    """
    fwd_cens, cc_cens = {}, {}
    dom2cat, cat_cens = {}, {}
    num_cc_cens, num_cc_access = collections.Counter(), collections.Counter()
    # Start by reading in all of the data
    with open(in_file, 'r') as file_p:
        reader = csv.DictReader(file_p)
        for row in reader:
            fwd = row['forwarder']
            domain = row['domain']
            country = row['fwd_cc']
            if country == "":
                country = "N/A"
            is_dns_censor = row["isDNSBlock"]
            if is_dns_censor == "True":
                is_dns_censor = True
            else:
                is_dns_censor = False
            if (domain is None) or (country is None):
                continue
            cats = row['categories'].split("|")
            entry = {'domain': domain, 'is_censored': is_dns_censor,
                     'categories': cats, 'country': country}

            if domain not in dom2cat:
                dom2cat[domain] = cats

            if fwd not in fwd_cens:
                fwd_cens[fwd] = {domain: entry}
            else:
                fwd_cens[fwd][domain] = entry

            if country not in cc_cens:
                cc_cens[country] = {}
            if domain not in cc_cens[country]:
                cc_cens[country][domain] = {'censored': 0, 'access': 0}
            if is_dns_censor:
                cc_cens[country][domain]['censored'] += 1
                num_cc_cens[country] += 1
            else:
                cc_cens[country][domain]['access'] += 1
                num_cc_access[country] += 1

            for cat in cats:
                if cat not in cat_cens:
                    cat_cens[cat] = {}
                if country not in cat_cens[cat]:
                    cat_cens[cat][country] = {'censored': 0, 'access': 0}
                if is_dns_censor:
                    cat_cens[cat][country]['censored'] += 1
                else:
                    cat_cens[cat][country]['access'] += 1

    countries = sorted(cc_cens.keys())
    num_cnt = len(countries)
    categories = sorted(cat_cens.keys())
    num_cat = len(categories)
    sites = sorted(dom2cat.keys())
    num_dom = len(sites)
    print "num countries: {} categories: {} sites: {}".format(num_cnt, num_cat,
                                                              num_dom)

    # print number of results per domain
    dom_counts = collections.Counter()
    for country in cc_cens:
        for domain in cc_cens[country]:
            dom_counts[domain] += cc_cens[country][domain]['access']
            dom_counts[domain] += cc_cens[country][domain]['censored']
    counts = [(domain, dom_counts[domain]) for domain in dom_counts]
    counts.sort(key=lambda ent: ent[1], reverse=True)
    with open(basename + 'site-result-count.txt', 'w') as out_p:
        for domain in dom_counts:
            out_p.write("{},{}\n".format(domain, dom_counts[domain]))

    # create lists of censorship by categories
    for cat in categories:
        cat_entries = []
        for country in countries:
            if country not in cat_cens[cat]:
                continue
            access = cat_cens[cat][country]['access']
            cens = cat_cens[cat][country]['censored']
            frac = float(cens) / float(access + cens)
            cat_entries.append((country, frac))
        cat_entries.sort(key=lambda ent: ent[1], reverse=True)
        with open(basename + cat + '-frac-censored-list.txt', 'w') as out_p:
            for entry in cat_entries:
                out_p.write("{},{}\n".format(entry[0], entry[1]))

    # create CDF and list of percent censored (overall) per country
    frac_cens_cnt, graph_data = [], {}
    for country in countries:
        total = float(num_cc_cens[country] + num_cc_access[country])
        frac = float(num_cc_cens[country]) / total
        frac_cens_cnt.append(frac)
        graph_data[country] = frac
    title = "CDF of fraction of domains censored per country"
    filename = basename + "country-frac-censored-cdf.pdf"
    xlabel = "Fraction of domains censored"
    ylabel = "Fraction of countries"
    create_cdf(frac_cens_cnt, xlabel, ylabel, title, filename,
               invert=True)

    # create the list of the most censored places in order
    cnt_frac_ordered = []
    for country in graph_data.keys():
        cnt_frac_ordered.append((country, graph_data[country]))
    cnt_frac_ordered.sort(key=lambda ent: ent[1], reverse=True)
    with open(basename + "country-frac-censored-list.txt", 'w') as out_p:
        for entry in cnt_frac_ordered:
            out_p.write("{},{}\n".format(entry[0], entry[1]))

    # plot the same country data on a world map
    title = "Heatmap of censorship worldwide"
    # create_world_map(graph_data, title)
    # custom bins in percents (divide by 100 to get fractions
    bins = [0, 0.008]
    create_world_map(graph_data, title, parts=bins, normalize=True)

    # create cdf of percent censored per site
    frac_cens_site, graph_data = [], {}
    missing = 0
    for site in sites:
        cens_fracs = []
        for country in cc_cens:
            cens, access = 0., 0.
            if site in cc_cens[country]:
                cens += cc_cens[country][site]['censored']
                access += cc_cens[country][site]['access']
                cens_fracs.append(cens / float(cens + access))
            else:
                missing += 1
        frac = stats.tmean(cens_fracs)
        frac_cens_site.append(frac)
        graph_data[site] = frac
    title = "CDF of censorship fractions for each site"
    filename = basename + "site-frac-censored-cdf.pdf"
    xlabel = "Mean censorship fraction for each country"
    ylabel = "Fraction of sites"
    print len(frac_cens_site), missing
    create_cdf(frac_cens_site, xlabel, ylabel, title, filename,
               invert=True)

    # create the list of the most censored sites in order
    site_frac_ordered = []
    for site in graph_data.keys():
        site_frac_ordered.append((site, graph_data[site]))
    site_frac_ordered.sort(key=lambda ent: ent[1], reverse=True)
    with open(basename + "site-frac-censored-list.txt", 'w') as out_p:
        for entry in site_frac_ordered:
            cats = "|".join(dom2cat[entry[0]])
            out_p.write("{},{},{}\n".format(entry[0], entry[1], cats))

    # start by constructing the data to graph
    graph_mask = [[True for x in range(num_cat)] for y in range(num_cnt)]
    graph_data = [[0 for x in range(num_cat)] for y in range(num_cnt)]
    # graph_mask = [[True for x in range(num_cat)] for y in range(50)]
    # graph_data = [[0 for x in range(num_cat)] for y in range(50)]
    for cat_indx in range(num_cat):
        # for cnt_indx in range(50):
        for cnt_indx in range(num_cnt):
            cnt = countries[cnt_indx]
            cat = categories[cat_indx]
            if (cat not in cat_cens) or (cnt not in cat_cens[cat]):
                continue
            cens = cat_cens[cat][cnt]['censored']
            access = cat_cens[cat][cnt]['access']
            tot = cens + access
            graph_data[cnt_indx][cat_indx] = float(cens) / float(tot)
            graph_mask[cnt_indx][cat_indx] = False
    graph_data = np.ma.array(graph_data, mask=graph_mask, dtype='float')
    title = ("Heat map of category censorship by country where the hue\n"
             "represents the extent of censorship")
    filename = basename + "cat-country-heatmap.pdf"

    create_heat_map(graph_data, categories, countries, title, filename)

    # create heat map of censorship (site vs country) w/
    # white/grey/red for access/block/inconclusive
    graph_mask = [[True for x in range(num_dom)] for y in range(num_cnt)]
    site_cnt = [[0 for x in range(num_dom)] for y in range(num_cnt)]
    for cnt_indx in range(len(countries)):
        cnt = countries[cnt_indx]
        for site_indx in range(len(sites)):
            site = sites[site_indx]
            if (cnt not in cc_cens) or (site not in cc_cens[cnt]):
                continue
            cens = cc_cens[cnt][site]['censored']
            access = cc_cens[cnt][site]['access']
            tot = cens + access
            site_cnt[cnt_indx][site_indx] = float(cens) / float(tot)
            graph_mask[cnt_indx][site_indx] = False
    site_cnt = np.ma.array(site_cnt, mask=graph_mask, dtype='float')

    # write out the matrix of countries and sites
    with open(dom_cnt_mat_file, 'w') as file_p:
        pickle.dump([countries, sites, site_cnt], file_p)

    # create the graph
    # title = ("Heat map of domain censorship by country where the hue\n"
    #          "represents the extent of censorship")
    # filename = basename + "dom-country-heatmap.pdf"
    # create_heat_map(site_cnt, sites, countries, title, filename,
    #                 size=(320, 24), font=8)

    # do PCA to find censorship themes

    # create pie chart of global and per country blocking for each category

    # create pie charts of censorship of local vs global content

    # create pie charts of censorship of locally relevant content vs
    # global content

    # types of censorship for site

    # prevalence of ip vs dns censorship per country


def create_world_map(data, title, normalize=False, parts=[]):
    """Create a URL to a colorized world map with the google maps API

    Params:
    normalize- scale the input so that all the fractions for plotting
        fall between 0 and 1. This is useful if your input is not a
        fraction between 0 and 1

    Note: you need to open this in a browser to plot

    Note: we expect the data as dictionary where the keys are two
    letter country codes and the values are numbers that we should
    plot with the heat map

    """
    # convert the dict to a list of tuples with country code and value
    outputs = data.items()
    # sort by value
    outputs = sorted(outputs, key=lambda entry: entry[1])
    min_v, max_v = float(outputs[0][1]), float(outputs[-1][1])
    print min_v, max_v
    if not normalize:
        min_v, max_v = 0., 1.
    if parts != []:
        min_v, max_v = parts

    # partition the space depending on the number of colors
    buckets = np.array([int(float(x[1] - min_v)/max_v * 5) for x in outputs])
    outputs = zip(buckets, *zip(*outputs))
    # parts = np.linspace(min_v * 100., (max_v - min_v) * 100. / max_v, 5)
    parts = np.linspace(min_v * 100., max_v * 100., num=6)
    labels = []
    for index in range(1, 6):
        # num1 = int(parts[index - 1])
        # num2 = int(parts[index])
        labels.append("{:.1f}-{:.1f}%".format(parts[index - 1],
                                              parts[index]))

    # define our color map (10 colors at most)
    # colors = ["fff7ec", "fee8c8", "fdd49e", "fdbb84", "fc8d59",
    #           "ef6548", "d7301f", "b30000", "7f0000", "000000"]
    #
    # use this map because it is colorblind friendly and photocopy
    # safe (it should still print in grayscale)
    # colors = ["ffffcc", "a1dab4", "41b6c4", "2c7fb8", "253494"]
    # this colorscheme should also print in grayscale
    # colors = ["fef0d9", "fdcc8a", "fc8d59", "d7301f"]
    # colors = ["c6dbef", "9ecae1", "6baed6", "3182bd", "08519c", "eff3ff"]
    colors = ["c6dbef", "9ecae1", "6baed6", "3182bd", "08519c"]

    # put all the country arguments together
    map_args = ["http://chart.apis.google.com/chart?cht=map:"
                "fixed=-60,-20,80,-35", "chs=600x400", "chma=0,60,0,0"]
    # set all the unselected countries to be gray
    cnts, clrs = ["AA"], ["808080", "808080"]
    # setup the colors for the legend
    for color in colors:
        cnts.append("AA")
        clrs.append(color)
    # add the countries
    for (color, country, val) in outputs:
        cnts.append(country.upper())
        if color >= len(colors):
            color = len(colors) - 1
        clrs.append(colors[color])
    map_args.append("chld=" + "|".join(cnts))
    map_args.append("chco=" + "|".join(clrs))

    # add the legend
    map_args.append("chdl=No+Data|" + "|".join(labels))
    map_args.append("chdls=000000,14")
    # add the background to make the graph more visible
    map_args.append("chf=bg,s,EFF3FF")

    # create the title
    title = title.replace(" ", "+")
    map_args.append("chtt=" + title)
    map_args.append("chts=000000,20,c")

    print "&".join(map_args)


def create_heat_map(data, x_labels, y_labels, title, filename, size=(8, 22),
                    font=None):
    fig, ax = plt.subplots()
    cmap = plt.cm.hot
    cmap.set_bad(color='grey')
    ax.pcolormesh(data, cmap=cmap)

    # replace kids and teens with just kids if it is in the labels
    for index in range(len(x_labels)):
        if x_labels[index] == "kids_and_teens":
            x_labels[index] = "kids"

    ax.set_frame_on(False)
    fig = plt.gcf()
    fig.set_size_inches(*size)

    # now setup the axes
    ax.invert_yaxis()
    # ax.xaxis.tick_top()
    # ax.set_xticklabels(x_labels, minor=False)
    # ax.set_yticklabels(y_labels, minor=False)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    if font is not None:
        ax.set_xticklabels(x_labels, minor=False, fontsize=font)
    else:
        ax.set_xticklabels(x_labels, minor=False)
    ax.set_yticklabels(y_labels, minor=False)
    plt.xticks(rotation=45, ha='right')

    # want a more natural, table-like display
    # ax.invert_yaxis()
    # x.xaxis.tick_top()

    # ax.xticks

    # setup a legend for the graph
    # cbar = plt.colorbar(cmap)
    # cbar.ax.get_yaxis().set_ticks([0.5, 2.5])
    # cbar.ax.set_yticklabels(["more censorship", "less censorship"])
    # cbar.ax.set_ylabel('Fraction of domains censored', rotation=270)

    # setup the legend for the graph by creating a 10x100 pixel colormap
    # fig = plt.gcf()
    # fig.set_size_inches(8, 26)
    # ax1 = plt.subplot(1, 5, 1)
    # ax1.set_frame_on(False)
    # cmap = plt.cm.hot
    # cmap.set_bad(color='grey')
    # ax1.pcolormesh(data, cmap=cmap)
    # ax1.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    # ax1.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
    # ax1.invert_yaxis()
    # ax1.set_xticklabels(x_labels, minor=False, rotation=45, ha='right')
    # ax1.set_yticklabels(y_labels, minor=False)
    # ax1.xticks(rotation=45, ha='right')

    # ax2 = plt.subplot(1, 5, 5)
    # increments = np.outer(np.arange(0, 1, 0.01), np.ones(10))
    # ax2.axis("off")
    # ax2.imshow(increments, aspect='auto', cmap=cmap, origin='lower')
    # # setup a legend for the graph
    # ax2.get_yaxis().set_ticks([0.5, 2.5])
    # ax2.set_yticklabels(["more censorship", "less censorship"])
    # ax2.set_ylabel('Fraction of domains censored', rotation=270)

    plt.title(title)
    plt.grid()
    plt.savefig(filename, fmt='pdf')


def create_pie_chart(data, title, filename, cutoff=0.01, verbose=False):
    """Create a pie chart from the given data

    Params:
    data- a collections.Counter where the keys should be the labels to
        the graph

    """
    total = float(sum(data.values()))
    frac_labels = []
    other_count = 0
    # compute the fraction for each label
    for cnt in data.keys():
        fraction = float(data[cnt]) / total
        if verbose:
            print "{}: {}".format(cnt, fraction)
        # if this slice is too small, just add it to other small
        # totals and display them together
        if fraction < cutoff:
            other_count += data[cnt]
            continue
        frac_labels.append((fraction, cnt))
    if other_count > 0:
        frac_labels.append((float(other_count) / total, "Other"))
    frac_labels.sort(key=lambda entry: entry[1])
    fractions, labels = zip(*frac_labels)
    plt.figure()
    plt.pie(fractions, labels=labels, autopct='%1.1f%%',
            colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))
    plt.title(title)
    plt.savefig(filename, fmt='pdf')


def create_cdf(data, x_label, y_label, title, filename, log=False,
               invert=False):
    """Given a set of data, create a CDF of the data, optionally in log
    format

    """
    data.sort()
    # create the y data for the CDF
    if invert:
        y_data = [float(x) / float(len(data)) for x in range(len(data), 0, -1)]
    else:
        y_data = [float(x) / float(len(data)) for x in range(len(data))]
    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if invert:
        ax.invert_xaxis()

    if log:
        plt.xscale('log')

    plt.title(title)
    plt.savefig(filename, fmt='pdf')


def update_good_data_for_raw_censorship(good_values, checkpoint):
    """Update the checkpoint file with the correct good data"""

    with open(good_values, 'r') as file_p:
        cdn_data = pickle.load(file_p)
        cdn_sites = cdn_data['cdnSites']

    with open(checkpoint, 'r') as file_p:
        stage, data = pickle.load(file_p)

    data1 = interference.get_http_responses(cdn_sites.keys())
    data2 = interference.get_http_responses(cdn_sites.keys())
    good_data = {}
    for site in data1.keys():
        good_data[site] = interferenceAnalysis.check_http_header_consistency(data1[site]['headers'], data2[site]['headers'])
        good_data[site].update(data1[site])

    data['good_data'] = good_data
    with open(checkpoint, 'w') as file_p:
        pickle.dump([stage, data], file_p)


def parse_args():
    parser = argparse.ArgumentParser(prog='analyzeDNS')
    subparsers = parser.add_subparsers(dest='cmd',
                                       help='Select a subcommand to run')

    # add the command line arguments for importing data
    import_help = ("Import the results files into a SQL db for easier "
                   "querying")
    imp = subparsers.add_parser('import', help=import_help)
    imp.add_argument('--domain', '-d', help="domainData.data file",
                     required=True)
    imp.add_argument('--forwarder', '-f', help="forwarderData.data file",
                     required=True)
    imp.add_argument('--raw', '-r', help="rawResp.data file",
                     required=True)
    imp.add_argument('--interest', '-i', help="file for interesting domains",
                     required=True)
    imp.add_argument('--db', '-b', help="output sqlite db file",
                     required=True)
    imp.add_argument('--dups', '-s', help="output duplicates to this file",
                     default=None)

    # add the command line arguments for finding duplicate responses
    dup_help = ("Find the duplicate responses with different answers")
    dup = subparsers.add_parser('dup', help=dup_help)
    dup.add_argument('--dup', '-d', help="duplicates.data file", required=True)
    dup.add_argument('--raw', '-r', help="rawResp.data file", required=True)
    int_help = "file to output complete duplicates"
    dup.add_argument('--output', '-o', help=int_help, required=True)

    # add the command line arguments for exploring duplicate responses
    dup_help = ("Explore the duplicate responses")
    dup = subparsers.add_parser('dup-ex', help=dup_help)
    dup.add_argument('--raw', '-r', help="duplicates.data file", required=True)
    int_help = "file for outputing interesting domains"
    dup.add_argument('--interest', '-i', help=int_help, required=True)
    control_help = "known good values for the resolved IPs and their ASes"
    dup.add_argument('--control', '-c', help=control_help,
                     default="../data/control-ases-ips-27-jan-2015.txt")

    # add the command line argument for characterizing the dataset
    char_help = ("Characterize the dataset")
    char = subparsers.add_parser('char', help=char_help)
    char.add_argument("--basename", '-b', required=True,
                      help="basename for output files")
    muts = char.add_mutually_exclusive_group()
    muts.add_argument("--db", '-d', help="*.db file with domain info")
    graph_help = "use the given data file to graph without recomputing"
    muts.add_argument("--graph-with", '-g', help=graph_help)

    # add the CLI for adding info to a censorship file
    info_help = ("Add more information to a file of censored domains")
    info = subparsers.add_parser('add-info', help=info_help)
    info.add_argument("--input", '-i', required=True,
                      help="file of just forwarders and censored domains")
    info.add_argument("--output", '-o', required=True,
                      help="file to output new data to")
    info.add_argument("--categories", '-c', help="file to get categories from",
                      default="../../utility/sites/domain-to-cat-26-jan-2015.txt")
    info.add_argument("--forwarders", '-f', help="forwarderData.data file",
                      default="../data/forwarderData.data")

    # add the CLI for creating the server fingerprints file
    ip_help = ("Create server fingerprinting analysis file")
    ip_fpr = subparsers.add_parser('ip-fpr', help=ip_help)
    ip_fpr.add_argument("--dns-db", '-d', required=True,
                        help="censorship-scan.db file with DNS scan info")
    ip_fpr.add_argument("--good-db", '-g', required=True,
                        help="db with known good HTTP headers")
    ip_fpr.add_argument("--http-db", required=True,
                        help="db with http scanning results")
    ip_fpr.add_argument("--output", '-o', required=True,
                        help="File to output fingerprints to as a CSV file")
    control_help = "known good values for the resolved IPs and their ASes"
    ip_fpr.add_argument('--control', '-c', help=control_help,
                        default="../data/control-ases-ips-27-jan-2015.txt")

    # add the CLI for creating the NXDOMAIN analysis file for what
    # resolvers do monetization
    nxdom = ("Create NXDOMAIN analysis file to show which resolvers are "
             "doing NXDOMAIN error monetization")
    nxdom = subparsers.add_parser('nxdomain', help=ip_help)
    raw_help = ("rawResp.data file from the www.cs.princeton.edu NXDOMAIN "
                "control run")
    nxdom.add_argument("--raw", '-r', required=True, help=raw_help)
    nxdom.add_argument("--output", '-o', required=True,
                       help="output file for monetization servers")

    # add the CLI for doing the summary analysis
    summary = ("Create summary analysis file saying whether or not each "
               "response is censored")
    summ = subparsers.add_parser('summary', help=summary)
    summ.add_argument('--raw', help="rawResp.data file",
                      required=True)
    summ.add_argument("--dns-db", required=True,
                      help="censorship-scan.db file with DNS scan info")
    # summ.add_argument("--good-db", required=True,
    #                   help="db with known good HTTP headers")
    summ.add_argument("--http-db", required=True,
                      help="db with http scanning results")
    summ.add_argument("--output", required=True,
                      help="File to output summarized results")
    summ.add_argument("--ptr", required=True,
                      help="PTR lookups for resolved IPs")
    summ.add_argument("--categories", help="file to get categories from",
                      required=True,
                      default="../../utility/sites/domain-to-cat-26-jan-2015.txt")
    control_help = "known good values for the resolved IPs and their ASes"
    summ.add_argument('--control', help=control_help, required=True,
                      default="../data/control-ases-ips-27-jan-2015.txt")
    mon_help = "Monetization control analysis results"
    summ.add_argument('--mon-control', help=mon_help, required=True,
                      default="../data/monetization-control-output.txt")

    # add the CLI for graphing the summary analysis
    graphs = ("Graph the data from the summary analysis")
    graph = subparsers.add_parser('graph', help=graphs)
    graph.add_argument('--summary', help="summary file of what is censored",
                       required=True)
    graph.add_argument('--basename', help="basename for outputting graphs",
                       required=True)

    update = subparsers.add_parser('update',
                                   help="update good http server data")
    good_help = "known good values for the resolved IPs and their ASes"
    update.add_argument('--good', '-g', required=True, help=good_help)
    update.add_argument('--checkpoint', '-c',
                        help="checkpointing file to update",
                        default='summary-analysis-checkpoint')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "import":
        import_data(args.domain, args.forwarder, args.raw, args.interest,
                    args.db, duplicates=args.dups)
    elif args.cmd == "dup":
        find_duplicates(args.dup, args.raw, args.output)
    elif args.cmd == "dup-ex":
        explore_duplicates(args.raw, args.interest, args.control)
    elif args.cmd == "char":
        if args.db:
            characterize_data(args.db, args.basename)
        elif args.graph_with:
            plot_char_data(args.basename, in_file=args.graph_with)
        else:
            raise Exception("Invalid option. You must select either "
                            "--graph-with or --db")
    elif args.cmd == "add-info":
        add_info_to_censorship_file(args.input, args.output, args.categories,
                                    args.forwarders)
    elif args.cmd == "ip-fpr":
        create_ip_fpr_analysis_file(args.output, args.good_db, args.http_db,
                                    args.dns_db, args.control)
    elif args.cmd == "nxdomain":
        find_monetization(args.raw, args.output)
    elif args.cmd == "summary":
        # find_censorship_raw(args.raw, args.dns_db, args.http_db,
        #                     args.good_db, args.categories,
        #                     args.mon_control, args.control,
        #                     args.output, args.ptr)
        find_censorship_raw(args.raw, args.dns_db, args.http_db,
                            args.categories, args.mon_control,
                            args.control, args.output, args.ptr)
    elif args.cmd == "graph":
        create_graphs(args.summary, args.basename)
    elif args.cmd == "update":
        update_good_data_for_raw_censorship(args.good, args.checkpoint)

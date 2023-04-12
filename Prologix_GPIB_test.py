from plx_gpib_ethernet import PrologixGPIBEthernet
import time

gpib = PrologixGPIBEthernet('10.1.11.155')

# open connection to Prologix GPIB-to-Ethernet adapter
gpib.connect()

# select gpib device at address 10
gpib.select(1)
gpib.set_timeout(2.9)

# send a query

result = gpib.query('*IDN?')
print(result)

# write without reading
# result = gpib.write('*RST')
# print(result)

# time.sleep(1)

result = gpib.query('INIT6:CONT?')
print(result)

result = gpib.query('FETC6:POW?')
print(result)


# close connection
gpib.close()
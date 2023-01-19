from plx_gpib_ethernet import PrologixGPIBEthernet

gpib = PrologixGPIBEthernet('10.1.11.58')

# open connection to Prologix GPIB-to-Ethernet adapter
gpib.connect()

# select gpib device at address 10
gpib.select(1)

# send a query
result = gpib.query('*IDN?')
print(result)
# => 'Stanford_Research_Systems,SR760,s/n41456,ver139\n'

# write without reading
result = gpib.write('*RST')
print(result)

# close connection
gpib.close()
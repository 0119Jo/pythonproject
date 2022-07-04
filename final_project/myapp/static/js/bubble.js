Highcharts.chart('container', {
  chart: {
    type: 'packedbubble',
    height: '100%'
  },
  title: {
    text: ' '
  },
  tooltip: {
    useHTML: true,
    pointFormat: '<b>{point.name}:</b> {point.value}건 사용됨'
  },
  plotOptions: {
    packedbubble: {
      minSize: '30%',
      maxSize: '120%',
      zMin: 0,
      zMax: 1000,
      layoutAlgorithm: {
        splitSeries: false,
        gravitationalConstant: 0.02
      },
      dataLabels: {
        enabled: true,
        format: '{point.name}',
        filter: {
          property: 'y',
          operator: '>',
          value: 250
        },
        style: {
          color: 'black',
          textOutline: 'none',
          fontWeight: 'normal'
        }
      }
    }
  },
  series: [{
    name: 'Europe',
    data: [{
      name: 'Germany',
      value: 767.1
    }, {
      name: 'Croatia',
      value: 20.7
    },
    {
      name: "Belgium",
      value: 97.2
    }]
  }, {
    name: 'Africa',
    data: [{
      name: "Senegal",
      value: 8.2
    },
    {
      name: "Cameroon",
      value: 9.2
    },
    {
      name: "Zimbabwe",
      value: 13.1
    }]
  }, {
    name: 'Oceania',
    data: [{
      name: "Australia",
      value: 409.4
    },
    {
      name: "New Zealand",
      value: 34.1
    },
    {
      name: "Papua New Guinea",
      value: 7.1
    }]
  }, {
    name: 'North America',
    data: [{
      name: "Costa Rica",
      value: 7.6
    },
    {
      name: "Honduras",
      value: 8.4
    },
    {
      name: "Jamaica",
      value: 8.3
    }]
  }, {
    name: 'South America',
    data: [{
      name: "El Salvador",
      value: 7.2
    },
    {
      name: "Uruguay",
      value: 8.1
    },
    {
      name: "Bolivia",
      value: 17.8
    },
    {
      name: "Trinidad and Tobago",
      value: 34
    }]
  }, {
    name: 'Asia',
    data: [{
      name: "Nepal",
      value: 6.5
    },
    {
      name: "Georgia",
      value: 6.5
    },
    {
      name: "Brunei Darussalam",
      value: 7.4
    }]
  }]
});
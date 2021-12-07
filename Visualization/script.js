let canvas = document.getElementById("rentchart");
var config = {
    type: "bar",
    data: {
        labels: ["Hammond Court", "Earlwood", "Meadow", "Ashvill", "Asafu"], 
        datasets: [{
            label: "Apartment Rent",
            data:[10, 19, 10, 25, 15],
            backgroundColor: [
                "rgba(255, 99, 132, 0.2)",
                "rgba(54, 162, 235, 0.2)",
                "rgba(255, 206, 86, 0.2)",
                "rgba(75, 192, 192, 0.2)",
                "rgba(153, 102, 255, 0.2)"
            ],
            borderColor: [
                "rgba(255,99,132,1)",
                "rgba(54, 162, 235, 1)",
                "rgba(255, 206, 86, 1)",
                "rgba(75, 192, 192, 1)",
                "rgba(153, 102, 255, 1)"
            ],
            borderWidth: 1
        }]
    },
}

var rentchart = new Chart(canvas, config);
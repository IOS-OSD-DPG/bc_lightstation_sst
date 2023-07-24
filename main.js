//Credit: James Hannah @cyborgsphinx

function changeProduct(event) {
	var i;
	var target = event.currentTarget;
	var product = target.id;

	clearContent("product")

	var showContent = document.getElementsByClassName(product);
	for (i = 0; i < showContent.length; i++) {
		showContent[i].style.display = "block";
	}
	target.className += " active";

	window.sessionStorage.setItem("product", product)
}

if (window.sessionStorage.getItem("product")) {
	document.getElementById(window.sessionStorage.getItem("product")).click();
}
//Credit: James Hannah @cyborgsphinx

//function changeAveraging(event) {
//	var i;
//	var target = event.currentTarget;
//	var product = target.id;
//
//	clearContent("product")
//
//	var showContent = document.getElementsByClassName(product);
//	for (i = 0; i < showContent.length; i++) {
//		showContent[i].style.display = "block";
//	}
//	target.className += " active";
//
//	window.sessionStorage.setItem("product", product)
//}
//
//if (window.sessionStorage.getItem("product")) {
//	document.getElementById(window.sessionStorage.getItem("product")).click();
//}

// Credit:
// https://www.w3schools.com/howto/howto_js_tabs.asp
// https://blog.hubspot.com/website/html-tabs
function openTab(evt, productName) {
  // Declare all variables
  var i, tabcontent, tablinks;

  // Get all elements with class="tabcontent" and hide them
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Get all elements with class="tablinks" and remove the class "active"
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  // Show the current tab, and add an "active" class to the button that opened the tab
  document.getElementById(productName).style.display = "block";
  evt.currentTarget.className += " active";

//  //   Get the element with id="defaultOpen" and click on it
//  document.getElementById("defaultOpen").click();
//  document.getElementsByClassName('tablinks')[0].click()
}
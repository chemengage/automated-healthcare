
// Switch between tabs (project details, team, inference)
function openCity(evt, tabname) {
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
    document.getElementById(tabname).style.display = "block";
    evt.currentTarget.className += " active";
    }
    document.getElementById("defaultOpen").click();


// Code to get the coordinates on mouse move over heatmap image
var elmids = ['hm_image'];
var x, y = 0;       // variables that will contain the coordinates

// Get X and Y position of the elm (from: vishalsays.wordpress.com)
function getXYpos(elm) {
x = elm.offsetLeft;        // set x to elm’s offsetLeft
y = elm.offsetTop;         // set y to elm’s offsetTop

elm = elm.offsetParent;    // set elm to its offsetParent

//use while loop to check if elm is null
// if not then add current elm’s offsetLeft to x
//offsetTop to y and set elm to its offsetParent
while(elm != null) {
    x = parseInt(x) + parseInt(elm.offsetLeft);
    y = parseInt(y) + parseInt(elm.offsetTop);
    elm = elm.offsetParent;
}
// returns an object with "xp" (Left), "=yp" (Top) position
return {'xp':x, 'yp':y};
}

// Get X, Y coords, and displays Mouse coordinates
function getCoords(e) {
  var xy_pos = getXYpos(this);

    x = e.pageX;
    y = e.pageY;
    x = x - xy_pos['xp'];
    y = y - xy_pos['yp'];

  // displays x and y coords in the #coords element
  //document.getElementById('coords').innerHTML = 'X= '+ x+ ' ,Y= ' +y;
}

// register onmousemove, and onclick the each element with ID stored in elmids
for(var i=0; i<elmids.length; i++) {
  if(document.getElementById(elmids[i])) {
    // calls the getCoords() function when mousemove
    document.getElementById(elmids[i]).onmousemove = getCoords;

    // execute a function when click
    document.getElementById(elmids[i]).onclick = function() {
      document.getElementById('regcoords').value = x+ ' , ' +y;
    };
  }
}

// Show readme file contents in the Project Description
$(function(){
  // Load `README.md` and show
  $("#included_content").load("./README.md", function(data) {
      marked.setOptions({
          gfm: true,
          tables: true,
          breaks: false,
          pedantic: false,
          sanitize: true,
          smartLists: true,
          smartypants: false,
          langPrefix: '',
          highlight: function(code, lang) {
              return code;
          }
      });
      var html_str = marked(data);
      $("#display_content").html(html_str);

      // Apply highlight.js again
      hljs.initHighlighting.called = false;
      hljs.initHighlighting();
  }); 
});
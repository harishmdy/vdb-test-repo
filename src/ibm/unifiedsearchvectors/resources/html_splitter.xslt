<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <!-- Copy all nodes and attributes by default -->
  <xsl:template match="@*|node()">
    <xsl:copy>
    <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>
  <!-- When matching class of runningheader do nothing.  
  This is an invisable header ex.  <span class="runningheader">  that is usualy is 99%  same as the header but comes before the header and ends up in a chunk by itelsef-->
    <xsl:template match="*[@class = 'runningheader']" />
    <xsl:template match="*[@class = 'runningfooter']" />

    <!-- some text that looks like graphics on top of page but comes before first header1 
        and ends up being a short chunk.  Just clean these out -->
    <xsl:template match="*[contains(@class,'st1')]" />
    <xsl:template match="*[contains(@class,'st2')]" />
    <xsl:template match="*[contains(@class,'st3')]" />

    <!-- clean out the links at the bottom of pages for navigation-->
    <xsl:template match="*[@class = 'familylinks']" />
    <xsl:template match="*[@class = 'parentlink']" />
    <xsl:template match="*[contains(@class,'linklist')]" />
    <xsl:template match="*[contains(@class,'relinfo')]" />
    <xsl:template match="*[contains(@class,'relconcepts')]" />
    <xsl:template match="*[contains(@class,'relref')]" />
    <!--xsl:template match="*[tokenize(@class,'\s')='linklist']" /-->

   <xsl:template match="*[normalize-space(.) = 'link']"/>

   <xsl:template match="script|style|comment()" /> <!-- make sure we remove all script nodes we dont' want those embedded --> 
    <!-- possibly put <i:aipgf id="adobe_illustrator_pgf" here to remove it from the final html -->
    <!---  like this <xsl:template match="*[@id = 'adobe_illustrator_pgf']" /> -->
    <!---  or maybe <xsl:template match="i:aipgf"] /> -->
    <!--<div class="familylinks"> <div class="parentlink"> linklist relinfo relconcepts -->
 
  <!-- take out newlines in any text nodes that aren't code -->
    <xsl:template match="text()[not(ancestor::code or codeblock)]">
      <xsl:value-of select="translate(.,'&#xA;',' ')"/>
    </xsl:template>
    <!--xsl:template match="text()[not(ancestor::codeph|systemoutput|screen|code|codeblock)]"-->
    <!-- codeph, systemoutput, screen, code, codeblock, <pre outputclass=screen>, maybe <pre class="codeblock"> -->
  

  <!-- this replaces any sub elements in the h1, gets rid of extra newlines and adds 2 newlines in front to favor splitting before an h1
    or example:
      <h2> first <span>second <div>third </div></span> fourth </h2>
    goes to this:
      <h2>

       first second third  fourth </h2>
  --> 
  <xsl:template match="h2">
    <h2>
    <xsl:text>&#xA;&#xA;</xsl:text> 
    <xsl:value-of select="translate(.,'&#xA;',' ')"/>
    <xsl:text> </xsl:text> 
    </h2>
  </xsl:template>

  <!-- put a single newline at the end of end of paragraph or li or table row or data definiation-->
  <xsl:template match="tr|li|p|dd"> 
      <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
      </xsl:copy>
      <xsl:text>&#xA; </xsl:text> 
  </xsl:template>

</xsl:stylesheet>




    